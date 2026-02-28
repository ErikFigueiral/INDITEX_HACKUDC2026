import os
import pandas as pd
from .config import DATA_DIR

def load_dataset(dataset_name="product_dataset"):
    csv_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)

    required_cols = ["product_asset_id", "product_image_url"]

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    return df[required_cols]