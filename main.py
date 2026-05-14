from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
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

#https://7896e90613zyun-8000.proxy.runpod.net
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
async def generate(request: Request, file: UploadFile = File(...)):

    try:
        uid = str(uuid.uuid4())

        ext = os.path.splitext(file.filename)[1].lower()

        # Validate file type
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Only PNG / JPG / JPEG / WEBP allowed"},
            )

        input_path = f"{UPLOAD_FOLDER}/{uid}{ext}"
        output_path = f"{OUTPUT_FOLDER}/{uid}.glb"

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate blueprint
        valid, msg = validate_blueprint(input_path)

        if not valid:
            os.remove(input_path)
            return JSONResponse(status_code=400, content={"error": msg})

        # Run conversion engine
        tool = FloorPlanConverter()
        result_path = tool.run(input_path, output_path)

        # Check output
        if not os.path.exists(result_path):
            return JSONResponse(
                status_code=500,
                content={"error": "GLB generation failed"}
            )

        # Remove input file after processing
        os.remove(input_path)

        # Generate dynamic base URL (works in Docker & cloud)
        PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

        if PUBLIC_BASE_URL:
           base_url = PUBLIC_BASE_URL.rstrip("/")
        else:
           base_url = str(request.base_url).rstrip("/")

        model_url = f"/files/{uid}.glb"
        full_url = f"{base_url}{model_url}"

        print("\n🔥 MODEL GENERATED:")
        print(full_url)

        return {
            "success": True,
            "model_url": model_url,   # for frontend
            "full_url": full_url      # optional (useful for testing)
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