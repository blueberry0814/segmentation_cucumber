import os
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

class DetectionSegmentationPipeline:
    def __init__(self, 
                 detection_checkpoint, 
                 sam_checkpoint, 
                 device=None, 
                 detection_score_threshold=0.5,
                 output_folder="./output_images"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.detection_model = self._load_detection_model(detection_checkpoint)
        self.detection_score_threshold = detection_score_threshold
        
        self.sam_model = self._load_sam_model(sam_checkpoint)
        self.sam_predictor = SamPredictor(self.sam_model)
        
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        json_folder = os.path.join(self.output_folder, "json")
        segmented_folder = os.path.join(self.output_folder, "segmented")
        os.makedirs(json_folder, exist_ok=True)
        os.makedirs(segmented_folder, exist_ok=True)
    
    def _load_detection_model(self, checkpoint_path):
        model = fasterrcnn_resnet50_fpn(num_classes=2)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _load_sam_model(self, checkpoint_path):
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return sam
    
    def process_image(self, image_path):
        print(f"Processing image: {image_path}")
        
        # Read image
        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
        
        # Detect objects
        with torch.no_grad():
            outputs = self.detection_model(img_tensor)
        
        # Parse detection outputs
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        # Print detection results
        print(f"Total detected objects: {len(boxes)}")
        print(f"Scores: {scores}")
        
        # Filter by score threshold
        selected_indices = scores >= self.detection_score_threshold
        selected_boxes = boxes[selected_indices]
        selected_scores = scores[selected_indices]
        selected_labels = labels[selected_indices]
        
        print(f"Objects after score threshold: {len(selected_boxes)}")
        
        # Convert to RGB for matplotlib and SAM
        img_rgb = np.array(img)
        
        # Initialize figure for visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_rgb)
        ax.axis('off')
        
        # Prepare SAM predictor
        self.sam_predictor.set_image(img_rgb)
        
        # Prepare data for JSON output
        bbox_data = []
        
        # Process each detected box
        for i, (box, score, label) in enumerate(zip(selected_boxes, selected_scores, selected_labels)):
            try:
                # Perform segmentation
                bbox_array = box.reshape(1, 4)
                masks, mask_scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox_array,
                    multimask_output=False,
                )
                
                if masks is not None and masks.size > 0:
                    # Draw mask and bounding box
                    self._draw_mask(masks[0], ax)
                    self._draw_bounding_box(img_rgb, box, ax)
                    
                    # Store bbox data
                    bbox_data.append({
                        "bbox": box.tolist(),
                        "score": float(score),
                        "label": int(label),
                        "mask_score": float(mask_scores[0]) if mask_scores is not None else 0.0
                    })
                    
            except Exception as e:
                print(f"Error processing box {i}: {e}")
        
        # Save JSON results
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.output_folder, "json", f"{base_filename}.json")
        with open(json_path, "w") as f:
            json.dump(bbox_data, f, indent=4)
        
        # Save annotated image
        plt.savefig(
            os.path.join(self.output_folder, "segmented", f"{base_filename}_segmented.jpg"), 
            bbox_inches='tight', 
            pad_inches=0
        )
        plt.close(fig)
        
        print(f"Processed {image_path}: {len(bbox_data)} objects detected and segmented")
    
    def _draw_bounding_box(self, image, bbox, ax, color='green'):
        """Draw bounding box on matplotlib axis"""
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2))
    
    def _draw_mask(self, mask, ax, random_color=False):
        """Draw segmentation mask on matplotlib axis"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])  # Dodger Blue with alpha
        
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def process_folder(self, image_folder):
        """
        Process all images in a folder
        
        Args:
            image_folder (str): Path to folder containing images
        """
        # Find all image files
        image_files = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))
        ]
        
        # Print image files found
        print(f"Found {len(image_files)} image files in {image_folder}")
        
        # Process each image
        for image_path in image_files:
            try:
                self.process_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def main():
    # Configuration
    detection_checkpoint = "./checkpoint/best_model.pth"
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    input_image_folder = "./test_images/"
    output_folder = "./output_images/"

    # Create pipeline
    pipeline = DetectionSegmentationPipeline(
        detection_checkpoint=detection_checkpoint,
        sam_checkpoint=sam_checkpoint,
        output_folder=output_folder,
        detection_score_threshold=0.5  # 낮출 수 있음
    )

    # Process images
    pipeline.process_folder(input_image_folder)

if __name__ == "__main__":
    main()