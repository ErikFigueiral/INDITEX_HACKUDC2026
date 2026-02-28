import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from .config import EMB_DIR
from .model_builder import build_embedding_model, IMG_SIZE
from .image_processing import download_image, pil_to_np, resize_to_square


def generate_embeddings(df, id_col, url_col, limit=None):

    if limit:
        df = df.head(limit)

    model = build_embedding_model()

    embeddings = []
    ids = []

    plot_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):

        try:
            img = download_image(row[url_col])
            img = pil_to_np(img)
            img = resize_to_square(img, IMG_SIZE)

            # 🔥 preprocess correcto para EfficientNet
            img = tf.keras.applications.efficientnet.preprocess_input(img)

            arr = np.expand_dims(img, axis=0)
            emb = model(arr, training=False).numpy()[0]

            embeddings.append(emb)
            ids.append(row[id_col])

            # 🔥 Plot solo los 2 primeros embeddings
            if plot_counter < 2:
                plt.figure(figsize=(10, 3))
                plt.plot(emb)
                plt.title(f"Embedding for ID {row[id_col]}")
                plt.show()
                plot_counter += 1

        except Exception as e:
            print("Error:", e)
            continue

    if len(embeddings) == 0:
        print("No embeddings generated.")
        return None

    embeddings = np.array(embeddings)
    ids = np.array(ids)

    np.save(f"{EMB_DIR}/embeddings.npy", embeddings)
    np.save(f"{EMB_DIR}/ids.npy", ids)

    print("Saved embeddings:", embeddings.shape)
    return embeddings