from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import numpy as np
from pathlib import Path

def load_and_preprocess(path: Path, size=(256, 256)) -> np.ndarray:
    """Load image -> float32 in [-1, 1], shape (1, H, W, C)."""
    img = load_img(str(path), target_size=size)
    arr = img_to_array(img)  # [0,255]
    arr = (arr - 127.5) / 127.5  # [-1,1]
    return np.expand_dims(arr.astype("float32"), axis=0)

def postprocess_to_uint8(x: np.ndarray) -> np.ndarray:
    """Map [-1,1] -> [0,255] uint8."""
    x = (x + 1.0) * 127.5
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def save_image(arr_uint8: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint8).save(str(out_path))
