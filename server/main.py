from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
from services.classifier import classifier_service
from models.model import HybridNeuMF

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STEP 1: LOAD DATA AND DEFINE CONSTANTS ---
try:
    with open("glasses_database.json", "r") as f:
        glasses_db = json.load(f)
    print(f" ✅ Loaded {len(glasses_db)} glasses from database.")
except FileNotFoundError:
    print(" ❌ ERROR: glasses_database.json not found!")
    glasses_db = {}

# These must be defined BEFORE the model initialization
NUM_SHAPES = 5
NUM_ITEMS = len(glasses_db) if glasses_db else 45
FACE_MAP = {"Heart": 0, "Oblong": 1, "Oval": 2, "Round": 3, "Square": 4}

# --- STEP 2: INITIALIZE THE MODEL ---
try:
    # Notice we use the 5-feature count to match your model.py
    model = HybridNeuMF(
        num_face_shapes=NUM_SHAPES, 
        num_items=NUM_ITEMS, 
        num_client_features=3, 
        num_item_features=5
    )
    model.load_state_dict(torch.load("spectacular_hybrid.pth", map_location=torch.device('cpu')))
    model.eval()
    print(" ✅ Model Loaded Successfully!")
except Exception as e:
    print(f" ❌ CRITICAL: Could not load model. Error: {e}")
    model = None

# --- STEP 3: ENDPOINTS ---

@app.post("/recommend")
async def recommend_glasses(
    file: UploadFile = File(...),
    features: str = Form(...) 
):
    if not model:
        raise HTTPException(status_code=500, detail="Recommendation model not loaded.")
    
    try:
        contents = await file.read()
        face_shape = classifier_service.predict(contents)
        shape_id = FACE_MAP.get(face_shape, 2) 

        try:
            feats_dict = json.loads(features)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON features format.")
            
        client_vector = [
            feats_dict.get("cheek_jaw_ratio", 1.0),
            feats_dict.get("face_hw_ratio", 1.0),
            feats_dict.get("midface_ratio", 1.0)
        ]

        current_num_items = len(glasses_db) 
        shape_tensor = torch.tensor([shape_id] * current_num_items)
        item_ids_tensor = torch.tensor(range(current_num_items))
        client_feat_tensor = torch.tensor([client_vector] * current_num_items, dtype=torch.float32)

        item_feats_list = []
        for i in range(current_num_items):
            g = glasses_db.get(str(i), {})
            item_feats_list.append([
                g.get("width", 0.0), 
                g.get("height", 0.0), 
                g.get("bridge_pos", 0.0),
                g.get("normalized_material", 0.0),
                g.get("normalized_rim", 0.0)
            ])
        item_feat_tensor = torch.tensor(item_feats_list, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(
                shape_tensor, 
                item_ids_tensor, 
                client_feat_tensor, 
                item_feat_tensor
            )

        scored_items = []
        for i, score in enumerate(predictions.tolist()):
            meta = glasses_db.get(str(i), {})
            scored_items.append({
                "glass_id": i,
                "name": meta.get("name", f"Model {i}"),
                "file_name": meta.get("file_name", ""),
                "score": round(score, 4)
            })
        
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        return {
            "status": "success",
            "detected_face_shape": face_shape, 
            "top_matches": scored_items[:5] 
        }

    except Exception as e:
        print(f"🔥 Recommendation Error: {e}")
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