import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import EMB_DIR
from .model_builder import EmbeddingModel
from .image_processing import download_image, preprocess_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings(df, id_col, url_col, limit=None):

    if limit:
        df = df.head(limit)

    model = EmbeddingModel(trainable=False).to(device)
    model.eval()

    embeddings = []
    ids = []

    plot_counter = 0

    with torch.no_grad():

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):

            try:
                img = download_image(row[url_col])
                tensor = preprocess_image(img).unsqueeze(0).to(device)

                emb = model(tensor).cpu().numpy()[0]

                embeddings.append(emb)
                ids.append(row[id_col])

                if plot_counter < 2:
                    plt.figure(figsize=(10, 3))
                    plt.plot(emb)
                    plt.title(f"Embedding ID {row[id_col]}")
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