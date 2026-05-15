from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests
import uuid
import os

from engine import FloorPlanConverter
from validator import validate_blueprint

app = FastAPI(title="Blueprint to 3D API")

# ---------------------------------------------------
# CORS (required for Flutter / mobile / web)
# ---------------------------------------------------
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


# ---------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------
class BlueprintRequest(BaseModel):
    image_url: str


# ---------------------------------------------------
# HOME ROUTE
# ---------------------------------------------------
@app.get("/")
def home():
    return {"status": "API Running"}


# ---------------------------------------------------
# GENERATE MODEL
# ---------------------------------------------------
@app.post("/generate")
async def generate(request: Request, data: BlueprintRequest):

    try:
        uid = str(uuid.uuid4())

        image_url = data.image_url

        input_path = f"{UPLOAD_FOLDER}/{uid}.png"
        output_path = f"{OUTPUT_FOLDER}/{uid}.glb"

        # =====================================================
        # DOWNLOAD IMAGE FROM FIREBASE URL
        # =====================================================

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

        # =====================================================
        # VALIDATE BLUEPRINT
        # =====================================================

        valid, msg = validate_blueprint(input_path)

        if not valid:

            if os.path.exists(input_path):
                os.remove(input_path)

            return JSONResponse(
                status_code=400,
                content={"error": msg},
            )

        # =====================================================
        # RUN CONVERSION ENGINE
        # =====================================================

        print("\n🔥 STARTING CONVERSION ENGINE")

        tool = FloorPlanConverter()

        result_path = tool.run(input_path, output_path)

        # =====================================================
        # CHECK OUTPUT
        # =====================================================

        if not os.path.exists(result_path):

            if os.path.exists(input_path):
                os.remove(input_path)

            return JSONResponse(
                status_code=500,
                content={
                    "error": "GLB generation failed"
                }
            )

        # =====================================================
        # CLEANUP INPUT IMAGE
        # =====================================================

        if os.path.exists(input_path):
            os.remove(input_path)

        # =====================================================
        # GENERATE MODEL URL
        # =====================================================

        PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

        if PUBLIC_BASE_URL:
            base_url = PUBLIC_BASE_URL.rstrip("/")
        else:
            base_url = str(request.base_url).rstrip("/")

        model_url = f"/files/{uid}.glb"
        full_url = f"{base_url}{model_url}"

        print("\n🔥 MODEL GENERATED:")
        print(full_url)

        # =====================================================
        # SUCCESS RESPONSE
        # =====================================================

        return {
            "success": True,
            "model_url": model_url,
            "full_url": full_url
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


# ---------------------------------------------------
# DOWNLOAD MODEL
# ---------------------------------------------------
@app.get("/files/{name}")
def get_file(name: str):

    path = os.path.join(OUTPUT_FOLDER, name)

    if not os.path.exists(path):
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"},
        )

    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=name,
    )