# server/services/classifier.py
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io

class FaceShapeClassifier:
    def __init__(self):
        print("   Loading Vision Model...")
        self.model_name = "metadome/face_shape_classification"
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            print("   ✅ Vision Model Loaded!")
        except Exception as e:
            print(f"   ❌ Vision Model Failed: {e}")
            self.model = None

    def predict(self, image_bytes):
        if not self.model:
            raise Exception("Model not loaded")

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]

# Create a singleton instance
# This runs once when the file is imported
classifier_service = FaceShapeClassifier()