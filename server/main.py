from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
from services.classifier import classifier_service
from model.model import HybridNeuMF

app = FastAPI()

# --- MIDDLEWARE ---
origins = ["http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD RESOURCES ---
NUM_SHAPES = 5
NUM_ITEMS = 50
FACE_MAP = {"Heart": 0, "Oblong": 1, "Oval": 2, "Round": 3, "Square": 4}

print(" 🧠 Loading Spectacular Hybrid Model...")
try:
    model = HybridNeuMF(NUM_SHAPES, NUM_ITEMS, num_geometric_features=3)
    model.load_state_dict(torch.load("spectacular_hybrid.pth"))
    model.eval()
    print(" ✅ Model Loaded Successfully!")
except Exception as e:
    print(f" ❌ CRITICAL: Could not load model. Error: {e}")
    model = None

@app.post("/recommend")
async def recommend_glasses(
    file: UploadFile = File(...),
    features: str = Form(...) # Receives JSON string like '{"cheek_jaw_ratio": 1.2, ...}'
):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 1. PHASE A: Computer Vision (ViT Classification)
        contents = await file.read()
        face_shape = classifier_service.predict(contents) # e.g., "Square"
        
        
        if face_shape not in FACE_MAP:
             # Fallback or Error
             print(f"Unknown shape {face_shape}, defaulting to Oval")
             face_shape = "Oval"

        # 2. PHASE B: Parse Geometric Features (Client Data)
        try:
            feats_dict = json.loads(features)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON features format.")
            
        geometric_vector = [
            feats_dict.get("cheek_jaw_ratio", 1.0),
            feats_dict.get("face_hw_ratio", 1.0),
            feats_dict.get("midface_ratio", 1.0)
        ]

        # 3. PHASE C: Hybrid Inference (Fusion)
        shape_id = FACE_MAP[face_shape]
        
        # Prepare Tensors for Batch Prediction (All 50 Items)
        shape_tensor = torch.tensor([shape_id] * NUM_ITEMS)
        item_tensor = torch.tensor(range(NUM_ITEMS))
        feature_tensor = torch.tensor([geometric_vector] * NUM_ITEMS, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(shape_tensor, item_tensor, feature_tensor)

        # 4. Ranking
        scored_items = []
        for i, score in enumerate(predictions.tolist()):
            scored_items.append({"glass_id": i, "score": score})
        
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        return {
            "status": "success",
            "detected_face_shape": face_shape, # Useful for UI feedback
            "method": "multimodal_hybrid_fusion",
            "recommendations": scored_items[:5]
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/classify-face")
async def classify_face_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        
        face_shape = classifier_service.predict(contents)
        
        return {
            "filename": file.filename,
            "face_shape": face_shape,
            "message": f"Successfully detected {face_shape} face."
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
