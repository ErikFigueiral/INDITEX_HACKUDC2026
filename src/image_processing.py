import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "Connection": "keep-alive",
}

def download_image(url: str, retries: int = 3, timeout: int = 12) -> Image.Image:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.0 + attempt * 0.5)
    raise last_err

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.uint8)

def resize_to_square(img_np: np.ndarray, size: int) -> np.ndarray:
    # simple resize; suficiente para hackathon (sin letterbox)
    img = Image.fromarray(img_np)
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32)