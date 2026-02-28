import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from .config import EMB_DIR, CLUSTER_DIR

def cluster_products(k=15):

    embeddings = np.load(f"{EMB_DIR}/embeddings.npy")
    ids = np.load(f"{EMB_DIR}/ids.npy")

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    df_out = pd.DataFrame({
        "product_asset_id": ids,
        "cluster": clusters
    })

    out_path = f"{CLUSTER_DIR}/clusters.csv"
    df_out.to_csv(out_path, index=False)

    score = silhouette_score(embeddings[:5000], clusters[:5000])
    print("Silhouette score:", score)
    print("Saved:", out_path)