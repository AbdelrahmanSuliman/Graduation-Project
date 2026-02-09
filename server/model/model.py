from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# 1. Load Model (Explicitly telling it to use ViT)
model_name = "metadome/face_shape_classification"
try:
    # We use ViTImageProcessor directly to bypass the "Unrecognized image processor" error
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Load and Preprocess Image
image_path = "round.jpg" # Make sure this matches your file
try:
    image = Image.open(image_path)
    # Ensure image is RGB (removes alpha channels if png)
    if image.mode != "RGB":
        image = image.convert("RGB")
except Exception:
    print("Error: Image not found.")
    exit()

# 3. Predict
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 4. Get Result
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

print("\n" + "="*30)
print(f" AI MODEL PREDICTION: {predicted_label}")
print("="*30 + "\n")