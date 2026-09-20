# BiRefNet Background Removal API (FastAPI + ONNX Runtime)

High-precision AI background removal microservice powered by **BiRefNet** and **ONNX Runtime**. Optimized for sub-pixel hair matting, edge decontamination, and streaming performance with zero disk I/O.

---

## Features

- **Sub-Pixel Hair Matting**: Preserves fine hair strands and semi-transparent edges without clipping.
- **Hair Color Decontamination**: Strips background color bleed and halo fringes around hair borders.
- **ONNX Runtime Engine**: Highly optimized CPU multi-threading and GPU (CUDA) support.
- **Zero Disk I/O Streaming**: Fast in-memory byte processing with FastAPI.
- **CORS Enabled**: Ready to connect with any web frontend (HTML/CSS/JS, React, Next.js, etc.).
- **Containerized**: Production-ready `Dockerfile` with automated ONNX model fetching.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` or `/api/health` | Health check, device info, and loaded model |
| `POST` | `/api/remove-bg` | Upload image (`multipart/form-data`) & receive transparent PNG |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/api/remove-bg?decontaminate=true" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  --output transparent.png
```

---

## Local Development

### Option 1: Running with Python

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *(The ONNX model ~927 MB will automatically download on first run if not already present in `models/`).*

### Option 2: Running with Docker

1. **Build image**:
   ```bash
   docker build -t birefnet-backend .
   ```

2. **Run container**:
   ```bash
   docker run -p 8000:8000 birefnet-backend
   ```

---

## Deploying to Azure Container Apps (Azure Portal GUI Console)

Follow these steps to deploy directly from the Azure Portal without using any CLI commands:

### Step 1: Create Azure Container Registry (ACR)
1. In the [Azure Portal](https://portal.azure.com), search for **Container Registries** and click **Create**.
2. Select your Subscription and Resource Group (or create a new one, e.g. `bgremover-rg`).
3. Enter a unique **Registry name** (e.g. `birefnetacr`) and set SKU to **Basic**.
4. Click **Review + Create**, then **Create**.
5. Once created, go to the registry resource, navigate to **Settings > Access keys**, and toggle **Admin user** to **Enabled**. Note the username and password.

### Step 2: Create Azure Container App
1. Search for **Container Apps** in the Azure Portal search bar and click **Create**.
2. **Basics Tab**:
   - **Container App Name**: `birefnet-backend`
   - **Container Apps Environment**: Click *Create new* (give it a name, e.g. `birefnet-env`, and choose Consumption tier).
3. **Container Tab**:
   - Uncheck *"Use quickstart image"*.
   - Name: `birefnet-app`
   - **Image source**: Select **Azure Container Registry** (or select **GitHub Actions** if connecting repository directly).
   - **CPU and Memory**: Select **1.0 vCPU, 2.0 GiB memory** (recommended for BiRefNet 1024x1024 inference).
4. **Ingress Tab**:
   - **Ingress**: Select **Enabled**.
   - **Ingress traffic**: Select **Accepting traffic from anywhere** (External).
   - **Target Port**: Enter **`8000`**.
5. Click **Review + Create**, then **Create**.

### Step 3: Enable CI/CD via GitHub in Portal GUI
1. Open your newly created Container App.
2. Under the left navigation menu, go to **Deployment > Continuous deployment**.
3. Sign in to your GitHub account and select:
   - **Organization / Account**: `Alee-Noor`
   - **Repository**: `advance-ex-love1-bgrem-backend`
   - **Branch**: `main`
   - **Build type**: **Dockerfile** (path: `Dockerfile`)
4. Select your **Azure Container Registry** created in Step 1.
5. Click **Save**. Azure will automatically commit a GitHub Actions workflow file to your repo and trigger a build & deploy pipeline!
