from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, storage

import requests
import uuid
import os
import traceback

from engine import FloorPlanConverter
from validator import validate_blueprint

# =========================
# FIREBASE INIT
# =========================

cred = credentials.Certificate("firebase-adminsdk.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'sitevision-d982f.firebasestorage.app'
    })

# =========================
# APP
# =========================

app = FastAPI(title="Blueprint to 3D API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class BlueprintRequest(BaseModel):
    image_url: str


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def home():
    return {"status": "API Running"}


# =========================
# MAIN PIPELINE
# =========================

@app.post("/generate")
async def generate(data: BlueprintRequest):

    uid = str(uuid.uuid4())

    input_path = f"{UPLOAD_FOLDER}/{uid}.png"
    output_path = f"{OUTPUT_FOLDER}/{uid}.glb"

    try:
        # =========================
        # DOWNLOAD IMAGE
        # =========================
        print("\n🔥 Downloading blueprint...")
        response = requests.get(data.image_url, timeout=30)

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to download blueprint"}
            )

        with open(input_path, "wb") as f:
            f.write(response.content)

        # =========================
        # VALIDATION
        # =========================
        valid, msg, confidence = validate_blueprint(input_path)

        print(f"Validation: {valid}, confidence={confidence}")
        print(f"Message: {msg}")

        # Reject clearly invalid images
        if not valid and confidence < 0.35:

            if os.path.exists(input_path):
               os.remove(input_path)

               return JSONResponse(
                status_code=400,
                  content={
                  "success": False,
                  "error": msg,
                  "confidence": confidence
                   }
                 )

# Allow weak blueprints but tell user in response/logs
        if not valid:
            print(f"⚠️ Weak blueprint accepted: {msg}")

        tool = FloorPlanConverter()
        result_path = tool.run(input_path, output_path)

        if not result_path or not os.path.exists(result_path):
            return JSONResponse(
                status_code=500,
                content={"error": "GLB generation failed"}
            )

        # =========================
        # CLEAN INPUT
        # =========================
        if os.path.exists(input_path):
            os.remove(input_path)

        # =========================
        # UPLOAD TO FIREBASE
        # =========================
        print("\n🔥 Uploading to Firebase...")

        bucket = storage.bucket()
        blob = bucket.blob(f"models/{uid}.glb")

        blob.upload_from_filename(output_path)
        blob.make_public()

        model_url = blob.public_url

        # =========================
        # CLEAN OUTPUT
        # =========================
        if os.path.exists(output_path):
            os.remove(output_path)

        return {
            "success": True,
            "model_url": model_url,
            "confidence": confidence
        }

    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )