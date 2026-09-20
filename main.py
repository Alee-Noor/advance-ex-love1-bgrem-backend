"""
FastAPI Background Removal Microservice
=======================================
Clean, minimal API layer for BiRefNet background removal.
Delegates all AI and image processing to 'remover-birefnet-inference.py'.
"""

import os
import importlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# Dynamically import remover-birefnet-inference module
remover = importlib.import_module("remover-birefnet-inference")


# ---------------------------------------------------------------------
# Application Lifespan: Pre-loads model into memory on server startup
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load BiRefNet model once into memory
    remover.load_model()
    yield
    print("[API] Background removal service shut down.")


# ---------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------
app = FastAPI(
    title="BiRefNet Background Removal API",
    description="High-precision background removal with sub-pixel hair matting.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for future HTML/CSS/JS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service and model status."""
    return {
        "status": "healthy",
        "service": "birefnet-bg-remover",
        "device": remover.get_device(),
        "model": remover._MODEL_NAME
    }


@app.post("/api/remove-bg", tags=["Inference"])
async def remove_background(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    decontaminate: bool = Query(True, description="Remove background color bleed from hair")
):
    """
    Remove background from uploaded image and stream back transparent PNG.
    """
    # 1. Validate file format
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is not a supported image format."
        )

    try:
        # 2. Read image bytes into memory (zero disk I/O)
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # 3. Perform background removal using remover-birefnet-inference
        _, png_bytes = remover.process_image(
            input_data=image_bytes,
            decontaminate=decontaminate
        )

        # 4. Stream back transparent PNG
        base_name = os.path.splitext(file.filename or "image")[0]
        output_filename = f"{base_name}_transparent.png"

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{output_filename}"',
                "Cache-Control": "no-cache"
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error processing image: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Run locally with auto-reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
