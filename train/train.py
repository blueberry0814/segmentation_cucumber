# python3 -m torch.distributed.launch --nproc_per_node=4 train2.py
import os
import torch
import json
from torch.utils.data import random_split, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset
import shutil
import matplotlib.pyplot as plt

# Initialize Distributed Training
dist.init_process_group(backend='nccl')  # NCCL backend for multi-GPU
rank = dist.get_rank()
device = torch.device(f'cuda:{rank}')

# TensorBoard Writer
if rank == 0:
    writer = SummaryWriter(log_dir="./runs/fasterrcnn")

class CustomDataset(Dataset):
    def __init__(self, json_dir, img_dir, transforms=None, resize=(300, 300)):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.transforms = transforms
        self.resize = resize
        self.data = self._load_annotations()

    def _load_annotations(self):
        annotations = []
        for file in os.listdir(self.json_dir):
            if file.endswith('.json'):
                json_path = os.path.join(self.json_dir, file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    img_info = data.get("description", {})
                    bboxes = data.get("result", [])

                    # Ensure required fields exist
                    if not img_info.get("image") or not img_info.get("width") or not img_info.get("height"):
                        print(f"Skipping {json_path}: Missing essential image info.")
                        continue
                    
                    annotation = {
                        "image_path": img_info["image"],
                        "original_width": img_info["width"],
                        "original_height": img_info["height"],
                        "bboxes": [
                            [bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]]
                            for bbox in bboxes if "x" in bbox and "y" in bbox and "w" in bbox and "h" in bbox
                        ]
                    }
                    annotations.append(annotation)
                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
        return annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annotation = self.data[idx]
        img_path = os.path.join(self.img_dir, annotation["image_path"])
        
        # Check if image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        
        # Original dimensions
        orig_width, orig_height = annotation["original_width"], annotation["original_height"]
        
        # Resize the image
        if self.resize:
            resize_width, resize_height = self.resize
            img = img.resize((resize_width, resize_height), Image.BILINEAR)

        # Scale bounding box coordinates
        scale_x = resize_width / orig_width
        scale_y = resize_height / orig_height

        resized_bboxes = [
            [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
            for bbox in annotation["bboxes"]
        ]

        if len(resized_bboxes) == 0:
            raise ValueError(f"No bounding boxes found for {img_path}")

        boxes = torch.tensor(resized_bboxes, dtype=torch.float32)
        labels = torch.tensor([1] * len(resized_bboxes), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

def save_test_dataset(test_dataset, test_img_dir, test_label_dir, json_dir, img_dir):
    """
    Save test dataset JSON and image files to specified directories.

    Args:
    - test_dataset: Dataset split containing test data.
    - test_img_dir: Directory to save test images.
    - test_label_dir: Directory to save test JSON files.
    - json_dir: Original JSON directory.
    - img_dir: Original image directory.
    """
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    for idx in range(len(test_dataset)):
        annotation = test_dataset.dataset.data[test_dataset.indices[idx]]
        img_src_path = os.path.join(img_dir, annotation["image_path"])
        json_src_path = os.path.join(json_dir, annotation["image_path"].replace('.jpg', '.json'))

        # Copy image
        img_dest_path = os.path.join(test_img_dir, annotation["image_path"])
        if os.path.exists(img_src_path):
            shutil.copy2(img_src_path, img_dest_path)
        else:
            print(f"Image not found: {img_src_path}")

        # Copy JSON
        json_dest_path = os.path.join(test_label_dir, annotation["image_path"].replace('.jpg', '.json'))
        if os.path.exists(json_src_path):
            shutil.copy2(json_src_path, json_dest_path)
        else:
            print(f"JSON not found: {json_src_path}")

def calculate_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])  # 수정: max -> min
    y2 = torch.min(box1[3], box2[3])  # 수정: max -> min

    intersection = torch.max(x2 - x1, torch.tensor(0.0)) * torch.max(y2 - y1, torch.tensor(0.0))

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else torch.tensor(0.0)


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, checkpoint_path)

# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh training.")
        return 0, float("inf")  # Start from epoch 0 and reset best_val_loss

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_loss']



# Dataset and DataLoader
json_dir = "./data2/label"
img_dir = "./data2/img"
test_img_dir = "./data2/test/img"
test_label_dir = "./data2/test/label"
checkpoint_dir = "./checkpoint/"
checkpoint_file = "./checkpoint/model_epoch_37.pth"
b_size = 16

os.makedirs(checkpoint_dir, exist_ok=True)
full_dataset = CustomDataset(json_dir=json_dir, img_dir=img_dir, transforms=F.to_tensor, resize=(128, 128))

train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

save_test_dataset(test_dataset, test_img_dir, test_label_dir, json_dir, img_dir)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=b_size, num_workers=8, pin_memory=True, sampler=train_sampler, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=b_size, num_workers=8, pin_memory=True, sampler=val_sampler, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = fasterrcnn_resnet50_fpn(num_classes=2, rpn_post_nms_top_n_train=100, rpn_post_nms_top_n_test=50)
model = model.to(device)
model = DDP(model, device_ids=[rank])

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training logic remains unchanged
# ...


# Loss Tracking
best_val_loss = float("inf")
patience = 3
early_stop_counter = 0
start_epoch = 0
start_epoch, best_val_loss = load_checkpoint(checkpoint_file, model, optimizer, scheduler)

for epoch in range(start_epoch, 100):  # Resume from the last epoch
    # Training Loop
    model.train()
    train_loss = 0.0
    for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)  # Loss dictionary
        losses = sum(loss for loss in loss_dict.values())  # Total loss
        train_loss += losses.item()
        
        losses.backward()
        optimizer.step()

        # Log training loss to TensorBoard
        if rank == 0:
            writer.add_scalar("Loss/Train", losses.item(), epoch * len(train_loader) + step)

    train_loss /= len(train_loader)
    
    # Validation Loop (IoU calculation)
    model.eval()
    total_iou = 0.0
    num_boxes = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Run inference
            outputs = model(images)
            
            # Calculate IoU
            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes']
                pred_boxes = output['boxes']
                
                for gt_box in gt_boxes:
                    best_iou = 0.0
                    for pred_box in pred_boxes:
                        iou = calculate_iou(gt_box, pred_box)
                        best_iou = max(best_iou, iou)
                    
                    total_iou += best_iou
                    num_boxes += 1

    avg_iou = total_iou / num_boxes if num_boxes > 0 else 0.0
    val_loss = 1 - avg_iou  # Lower IoU corresponds to higher "loss"

    if rank == 0:
        writer.add_scalar("Loss/Validation", val_loss, epoch)

    # Save checkpoint every epoch
    if rank == 0:
        # Save checkpoint every epoch
        save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss, f"{checkpoint_dir}model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Avg IoU: {avg_iou:.4f}")
        
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        if rank == 0:
            save_checkpoint(epoch + 1, model, optimizer, scheduler, best_val_loss, f"{checkpoint_dir}best_model.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break
    
    scheduler.step()

if rank == 0:
    writer.close()

print("Training completed.")