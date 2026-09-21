import os
import sys
import io
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional


import threading

# Global ONNX session cache & thread synchronization
_GLOBAL_SESSION = None
_SESSION_LOCK = threading.Lock()
_SESSION_STATUS = "idle"  # "idle", "loading", "ready", "error"
_SESSION_ERROR = None
_MODEL_NAME = "BiRefNet-general"
_MODEL_FILENAME = "birefnet-general.onnx"
_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"
_TARGET_SIZE = 1024

# Normalization constants (ImageNet standard matching BiRefNet training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_status() -> str:
    """Returns current model status: 'idle', 'loading', 'ready', or 'error'."""
    return _SESSION_STATUS


def is_ready() -> bool:
    """Returns True if the ONNX session is initialized and ready for inference."""
    return _GLOBAL_SESSION is not None


def get_model_path() -> str:
    """Finds the ONNX model file locally in ./models/ or system cache."""
    # 1. Check local models/ folder inside backend
    local_path = Path(__file__).parent / "models" / _MODEL_FILENAME
    if local_path.exists():
        return str(local_path)

    # 2. Check user's .u2net cache directory
    u2net_path = Path.home() / ".u2net" / _MODEL_FILENAME
    if u2net_path.exists():
        return str(u2net_path)

    # Fallback to local file in models directory
    return str(local_path)


def get_device() -> str:
    """Returns available execution providers in ONNX Runtime."""
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "GPU (CUDAExecutionProvider)"
    return "CPU (CPUExecutionProvider)"


def load_model():
    """
    Initializes and warms up the ONNX Runtime InferenceSession in a thread-safe manner.
    Auto-downloads the model if not present.
    """
    global _GLOBAL_SESSION, _SESSION_STATUS, _SESSION_ERROR
    if _GLOBAL_SESSION is not None:
        return _GLOBAL_SESSION

    with _SESSION_LOCK:
        if _GLOBAL_SESSION is not None:
            return _GLOBAL_SESSION

        _SESSION_STATUS = "loading"
        try:
            model_path = get_model_path()
            if not os.path.exists(model_path):
                print(f"--- BiRefNet ONNX model not found locally at {model_path} ---")
                print(f"--- Downloading from {_MODEL_URL} (~927 MB)... ---")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                import urllib.request
                def _reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        if count % 2000 == 0:
                            sys.stdout.write(f"\rDownloading model: {percent}%")
                            sys.stdout.flush()
                urllib.request.urlretrieve(_MODEL_URL, model_path, _reporthook)
                print("\n--- Model download complete! ---")

            # Configure execution providers (CUDA if available, otherwise CPU)
            available = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

            # Session options for multi-threaded CPU performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = os.cpu_count() or 4

            print(f"--- Loading BiRefNet ONNX Model ({model_path}) ---")
            print(f"--- Using Provider: {providers[0]} ---")
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            _GLOBAL_SESSION = session
            _SESSION_STATUS = "ready"
            print("--- BiRefNet ONNX engine is ready for inference ---")
            return _GLOBAL_SESSION
        except Exception as e:
            _SESSION_STATUS = "error"
            _SESSION_ERROR = str(e)
            print(f"--- Failed to load BiRefNet model: {e} ---")
            raise


def decontaminate_hair_color(image_rgb: np.ndarray, alpha_float: np.ndarray, bg_blur_radius: int = 31) -> np.ndarray:
    """
    Strips background color bleed from semi-transparent hair strands
    to eliminate fringes and halos when placed on new backgrounds.
    Exact implementation from test1.py.
    """
    img_f = image_rgb.astype(np.float32)
    alpha = np.clip(alpha_float, 0.0, 1.0)

    inv_alpha = 1.0 - alpha
    weight = inv_alpha + 1e-5
    bg_sum = cv2.blur(img_f * weight[:, :, None], (bg_blur_radius, bg_blur_radius))
    w_sum = cv2.blur(weight, (bg_blur_radius, bg_blur_radius)) + 1e-5
    bg_est = bg_sum / w_sum[:, :, None]

    clean_rgb = img_f.copy()
    semi_mask = (alpha > 0.03) & (alpha < 0.97)

    if np.any(semi_mask):
        a_exp = alpha[semi_mask, None]
        bg_semi = bg_est[semi_mask]
        cleaned = (img_f[semi_mask] - (1.0 - a_exp) * bg_semi) / np.maximum(a_exp, 0.30)
        clean_rgb[semi_mask] = np.clip(cleaned, 0.0, 255.0)

    return clean_rgb.astype(np.uint8)


def process_image(
    input_data: Union[str, bytes, io.BytesIO, Image.Image],
    decontaminate: bool = True
) -> Tuple[Image.Image, bytes]:
    """
    High-performance ONNX in-memory inference function.
    Accepts bytes, PIL Image, or file path and returns (PIL_Image, PNG_bytes).
    Zero disk I/O, optimized for FastAPI streaming.
    """
    session = load_model()

    if isinstance(input_data, bytes):
        orig_image = Image.open(io.BytesIO(input_data)).convert("RGB")
    elif isinstance(input_data, io.BytesIO):
        orig_image = Image.open(input_data).convert("RGB")
    elif isinstance(input_data, Image.Image):
        orig_image = input_data.convert("RGB")
    elif isinstance(input_data, str):
        orig_image = Image.open(input_data).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(input_data)}")

    orig_w, orig_h = orig_image.size

    # 1. Native 1024x1024 Preprocessing
    img_1024 = orig_image.resize((_TARGET_SIZE, _TARGET_SIZE), Image.BILINEAR)
    arr = np.array(img_1024).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    input_tensor = np.expand_dims(arr, 0)  # (1, 3, 1024, 1024)

    # 2. ONNX Inference
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: input_tensor})[0]

    # Sigmoid activation to convert logits to probabilities (0.0 .. 1.0)
    alpha_1024 = 1.0 / (1.0 + np.exp(-raw_output[0, 0]))

    # 3. Resize continuous alpha back to full native resolution
    alpha_pil = Image.fromarray((alpha_1024 * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.LANCZOS)
    alpha_float = np.array(alpha_pil).astype(np.float32) / 255.0

    # 4. Hair color decontamination (anti-halo)
    if decontaminate:
        clean_rgb = decontaminate_hair_color(np.array(orig_image), alpha_float)
    else:
        clean_rgb = np.array(orig_image)

    # 5. Composite final RGBA
    final_output = Image.fromarray(clean_rgb).convert("RGBA")
    final_output.putalpha(alpha_pil)

    # In-memory PNG byte streaming
    buffer = io.BytesIO()
    final_output.save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()

    return final_output, png_bytes


def remove_background_birefnet(input_path: str, output_path: str, target_size: int = 1024):
    """
    Standalone file-to-file function matching test1.py interface.
    """
    print("Step 1: Generating sub-pixel alpha matte via ONNX...")
    final_output, _ = process_image(input_path, decontaminate=True)

    print("Step 2: Saving transparent PNG...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    final_output.save(output_path, "PNG")
    print(f"--- Done! Transparent PNG saved to: {output_path} ---")


# =====================================================================
if __name__ == "__main__":
    # =================================================================
    # >>> SPECIFY YOUR INPUT & OUTPUT PATHS HERE <<<
    # =================================================================
    INPUT = ""                         # <-- Put your image path here (e.g. "girl.jpg", "test2.jpg")
    OUTPUT = "birefnet_output.png"     # <-- Output transparent PNG path
    # =================================================================

    # Also allows passing input path via command line: python remover-birefnet-inference.py image.jpg
    if len(sys.argv) > 1 and sys.argv[1].strip():
        INPUT = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].strip():
        OUTPUT = sys.argv[2]

    if not INPUT or not INPUT.strip():
        print("=" * 65)
        print("  NOTICE: No input path specified.")
        print('  Please open remover-birefnet-inference.py and set INPUT = "your_image.jpg"')
        print("  Or run: python remover-birefnet-inference.py your_image.jpg")
        print("=" * 65)
        sys.exit(0)

    if not os.path.exists(INPUT):
        print(f"Error: Input file '{INPUT}' does not exist.")
        sys.exit(1)

    try:
        remove_background_birefnet(INPUT, OUTPUT)
    except Exception as e:
        print(f"An error occurred: {e}")