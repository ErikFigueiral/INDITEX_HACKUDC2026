import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")
EMB_DIR = os.path.join(OUT_DIR, "embeddings")
CLUSTER_DIR = os.path.join(OUT_DIR, "clusters")
MODEL_DIR = os.path.join(OUT_DIR, "models")

for d in [OUT_DIR, EMB_DIR, CLUSTER_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)