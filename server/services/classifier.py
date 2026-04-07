# server/services/classifier.py
from transformers import ViTImageProcessor, ViTForImageClassification
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import io
import os

class FaceShapeClassifier:
    def __init__(self):
        print("🚀 Initializing AI Vision Pipeline...")
        
        self.mtcnn = MTCNN(keep_all=False, margin=20, device='cpu') 
        
        # 2. Load the Vision Transformer (ViT) from HuggingFace
        self.model_name = "metadome/face_shape_classification"
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            print("✅ Vision Transformer Loaded (metadome/face_shape_classification)")
        except Exception as e:
            print(f"❌ Failed to load ViT model: {e}")
            self.model = None

    def predict(self, image_bytes: bytes) -> str:
        """
        Processes an image by detecting the face, applying a soft square crop,
        and classifying the face shape.
        """
        if not self.model:
            raise Exception("Classifier Model not initialized properly.")

        # Load image and ensure it is RGB (removes Alpha channel from PNGs)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes, _ = self.mtcnn.detect(image)
        
        processed_image = image # Fallback to original if detection fails
        
        if boxes is not None:
            box = boxes[0] # Coordinates of the most prominent face [x1, y1, x2, y2]
            
            # This prevents vertical stretching which causes "Oblong" hallucinations
            w = box[2] - box[0]
            h = box[3] - box[1]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            # MARGIN CONTROL: 1.4 expands the box by 40% to include hair and ears
            # Lower this (e.g., 1.2) for a tighter crop, increase for a looser one.
            margin_factor = 1.2
            side = max(w, h) * margin_factor
            
            img_w, img_h = image.size
            new_box = [
                max(0, cx - side/2), 
                max(0, cy - side/2), 
                min(img_w, cx + side/2), 
                min(img_h, cy + side/2)
            ]
            
            processed_image = image.crop(new_box)
            
        inputs = self.processor(images=processed_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        raw_label = self.model.config.id2label[predicted_class_idx]
        
        final_shape = raw_label.capitalize()
        print(f"🎯 AI Prediction: {final_shape}")
        
        return final_shape

classifier_service = FaceShapeClassifier()