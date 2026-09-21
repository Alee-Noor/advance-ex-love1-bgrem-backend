# Ultra-lightweight Python base image
FROM python:3.10-slim

# System dependencies for OpenCV image processing and curl for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight dependencies (ONNX Runtime, FastAPI, OpenCV, Pillow)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

# Ensure models directory exists and download ONNX model if missing
RUN mkdir -p /app/models && \
    if [ ! -f /app/models/birefnet-general.onnx ]; then \
        echo "Downloading BiRefNet ONNX model (927MB)..." && \
        curl -L -f --retry 3 -o /app/models/birefnet-general.onnx \
        https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx ; \
    fi

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI via Uvicorn with dynamic port support (defaults to 8000)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

