import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    )
}


def download_image(url: str, retries: int = 3, timeout: int = 12):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise last_err


def pil_to_np(img):
    return np.asarray(img).astype(np.uint8)


def resize_to_square(img_np, size):
    img = Image.fromarray(img_np)
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32)


def preprocess_image(img_np, size):
    img = resize_to_square(img_np, size)
    img = keras.applications.efficientnet.preprocess_input(img)
    return img


def plot_image(img_np, title=None):
    plt.imshow(img_np.astype(np.uint8))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()