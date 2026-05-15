from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, storage

import requests
import uuid
import os

from engine import FloorPlanConverter
from validator import validate_blueprint

# =====================================================
# FIREBASE ADMIN SETUP
# =====================================================

cred = credentials.Certificate("firebase-adminsdk.json")

firebase_admin.initialize_app(cred, {
    'storageBucket': 'sitevision-d982f.firebasestorage.app'
})

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="Blueprint to 3D API")

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# FOLDERS
# =====================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# REQUEST MODEL
# =====================================================

class BlueprintRequest(BaseModel):
    image_url: str

# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")
def home():
    return {"status": "API Running"}

# =====================================================
# GENERATE MODEL
# =====================================================

@app.post("/generate")
async def generate(request: Request, data: BlueprintRequest):

    try:

        uid = str(uuid.uuid4())

        image_url = data.image_url

        input_path = f"{UPLOAD_FOLDER}/{uid}.png"
        output_path = f"{OUTPUT_FOLDER}/{uid}.glb"

        # =================================================
        # DOWNLOAD IMAGE FROM FIREBASE
        # =================================================

        print("\n🔥 DOWNLOADING IMAGE:")
        print(image_url)

        response = requests.get(image_url)

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Failed to download blueprint image"
                },
            )

        with open(input_path, "wb") as f:
            f.write(response.content)

        print("\n🔥 IMAGE DOWNLOADED SUCCESSFULLY")

        # =================================================
        # VALIDATE BLUEPRINT
        # =================================================

        valid, msg = validate_blueprint(input_path)

        if not valid:

            if os.path.exists(input_path):
                os.remove(input_path)

            return JSONResponse(
                status_code=400,
                content={"error": msg},
            )

        # =================================================
        # RUN CONVERSION ENGINE
        # =================================================

        print("\n🔥 STARTING CONVERSION ENGINE")

        tool = FloorPlanConverter()

        result_path = tool.run(input_path, output_path)

        # =================================================
        # CHECK OUTPUT
        # =================================================

        if not os.path.exists(result_path):

            if os.path.exists(input_path):
                os.remove(input_path)

            return JSONResponse(
                status_code=500,
                content={
                    "error": "GLB generation failed"
                }
            )

        # =================================================
        # CLEANUP INPUT IMAGE
        # =================================================

        if os.path.exists(input_path):
            os.remove(input_path)

        # =================================================
        # UPLOAD GLB TO FIREBASE STORAGE
        # =================================================

        print("\n🔥 UPLOADING GLB TO FIREBASE")

        bucket = storage.bucket()

        blob = bucket.blob(f"models/{uid}.glb")

        blob.upload_from_filename(output_path)

        blob.make_public()

        firebase_model_url = blob.public_url

        print("\n🔥 FIREBASE MODEL URL:")
        print(firebase_model_url)

        # =================================================
        # OPTIONAL CLEANUP OUTPUT FILE
        # =================================================

        if os.path.exists(output_path):
            os.remove(output_path)

        # =================================================
        # SUCCESS RESPONSE
        # =================================================

        return {
            "success": True,
            "model_url": firebase_model_url
        }

    except Exception as e:

        import traceback

        print("\n❌ ERROR TRACEBACK:\n")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            },
        )