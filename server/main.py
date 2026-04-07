from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import os
from services.classifier import classifier_service
from models.model import HybridNeuMF
from pydantic import BaseModel
from database.database import save_user_feedback, init_db

app = FastAPI()

# Enable CORS for frontend integration
origins = ["http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_db()

# --- DATABASE LOADING ---
db_path = "./database/glasses_database.json"
try:
    with open(db_path, "r") as f:
        glasses_db = json.load(f)
    print(f" ✅ Loaded {len(glasses_db)} glasses from database.")
except FileNotFoundError:
    print(f" ❌ ERROR: {db_path} not found!")
    glasses_db = {}

# --- MODEL CONFIGURATION ---
NUM_SHAPES = 5
NUM_ITEMS = len(glasses_db) if glasses_db else 45
# Client features = 4 (cheek_jaw, face_hw, midface + width_diff engineered feature)
NUM_CLIENT_FEATURES = 4 
NUM_ITEM_FEATURES = 5
FACTOR_NUM = 32
FACE_MAP = {"Heart": 0, "Oblong": 1, "Oval": 2, "Round": 3, "Square": 4}

class FeedbackRequest(BaseModel):
    glass_id: int
    detected_face_shape: int
    features: list[float]
    liked: bool

# --- MODEL LOADING ---
try:
    model = HybridNeuMF(
        num_face_shapes=NUM_SHAPES, 
        num_items=NUM_ITEMS, 
        num_client_features=NUM_CLIENT_FEATURES, 
        num_item_features=NUM_ITEM_FEATURES,
        factor_num=FACTOR_NUM
    )
    # Load weights trained with BCELoss
    model.load_state_dict(torch.load("spectacular_hybrid.pth", map_location=torch.device('cpu')))
    model.eval()
    print(" ✅ Hybrid NeuMF Model Loaded Successfully!")
except Exception as e:
    print(f" ❌ CRITICAL: Could not load model. Error: {e}")
    model = None

@app.post("/recommend")
async def recommend_glasses(
    file: UploadFile = File(...),
    features: str = Form(...) 
):
    if not model:
        raise HTTPException(status_code=500, detail="Recommendation model not loaded.")
    
    try:
        # 1. Detect Face Shape
        contents = await file.read()
        face_shape = classifier_service.predict(contents)
        shape_id = FACE_MAP.get(face_shape, 2) 

        # 2. Parse Face Features
        try:
            feats_dict = json.loads(features)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON features format.")
            
        base_client_vector = [
            feats_dict.get("cheek_jaw_ratio", 1.0),
            feats_dict.get("face_hw_ratio", 1.0),
            feats_dict.get("midface_ratio", 1.0)
        ]

        # 3. Prepare Tensors
        current_num_items = len(glasses_db) 
        shape_tensor = torch.tensor([shape_id] * current_num_items)
        item_ids_tensor = torch.tensor(range(current_num_items))
        
        item_feats_list = []
        engineered_client_feats = []

        for i in range(current_num_items):
            g = glasses_db.get(str(i), {})
            g_width = g.get("width", 0.5)
            item_feats_list.append([
                g_width, g.get("height", 0.5), g.get("bridge_pos", 0.5),
                g.get("normalized_material", 0.0), g.get("normalized_rim", 0.0)
            ])
            width_diff = abs(base_client_vector[1] - g_width)
            engineered_client_feats.append(base_client_vector + [width_diff])

        client_feat_tensor = torch.tensor(engineered_client_feats, dtype=torch.float32)
        item_feat_tensor = torch.tensor(item_feats_list, dtype=torch.float32)

        # 4. Generate Predictions
        with torch.no_grad():
            predictions = model(shape_tensor, item_ids_tensor, client_feat_tensor, item_feat_tensor)

        # 5. Format and Sort All Results
        all_scored_items = []
        for i, score in enumerate(predictions.tolist()):
            meta = glasses_db.get(str(i), {})
            all_scored_items.append({
                "glass_id": i,
                "name": meta.get("name", f"Model {i}"),
                "file_name": meta.get("file_name", ""),
                "score": round(score, 4)
            })
        
        # Sort by score descending (Highest to Lowest)
        all_scored_items.sort(key=lambda x: x["score"], reverse=True)

        top_5 = all_scored_items[:5]
        bottom_5 = all_scored_items[-5:] # Last 5 items in the sorted list

        # Console Logs for debugging
        print(f"\n--- RECOMMENDING FOR {face_shape.upper()} FACE ---")
        print(f"TOP 5 SCORES: {[x['score'] for x in top_5]}")
        print(f"LOWEST 5 SCORES: {[x['score'] for x in bottom_5]}")

        return {
            "status": "success",
            "detected_face_shape": face_shape, 
            "top_matches": top_5,
            "lowest_matches": bottom_5, # New field in JSON response
            "detected_face_shape_id": shape_id
        }

    except Exception as e:
        print(f"Recommendation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/feedback")
async def post_feedback(request: FeedbackRequest):
    save_user_feedback(
        glass_id=request.glass_id,
        shape_id=request.detected_face_shape,
        features=request.features,
        liked=request.liked
    )
    return {"status": "success"}