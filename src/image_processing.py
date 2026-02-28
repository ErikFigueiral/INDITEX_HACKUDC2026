import time
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from torchvision import transforms


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def download_image(url, retries=3, timeout=12):
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise last_err


def preprocess_image(img):
    return transform(img)


def plot_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()