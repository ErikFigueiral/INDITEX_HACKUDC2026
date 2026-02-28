import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import EMB_DIR
from .model_builder import build_embedding_model, IMG_SIZE
from .image_processing import download_image, pil_to_np, resize_to_square

def generate_embeddings(df, id_col, url_col, limit=None):

    if limit:
        df = df.head(limit)

    model = build_embedding_model(trainable=False)

    embeddings = []
    ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        try:
            img = download_image(row[url_col])
            img = pil_to_np(img)
            img = resize_to_square(img, IMG_SIZE)

            arr = np.expand_dims(img, axis=0)
            emb = model(arr, training=False).numpy()[0]

            embeddings.append(emb)
            ids.append(row[id_col])

        except Exception as e:
            print("Error:", e)
            continue

    if not embeddings:
        print("No embeddings generated.")
        return None

    embeddings = np.array(embeddings)

    np.save(f"{EMB_DIR}/embeddings.npy", embeddings)
    np.save(f"{EMB_DIR}/ids.npy", np.array(ids))

    print("Saved embeddings:", embeddings.shape)
    return embeddings