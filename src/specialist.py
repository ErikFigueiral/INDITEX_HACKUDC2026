import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from .segmentation import PersonSegmenter
from .image_processing import download_image, pil_to_np, resize_to_square
from .model_builder import IMG_SIZE
from .config import OUT_DIR


# --------------------------
# Modelo especialista cluster
# --------------------------

def build_specialist_model():

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = False

    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))

    # 🔥 AUGMENTATION
    x = tf.keras.layers.RandomFlip("horizontal")(inp)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomContrast(0.1)(x)

    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    model = tf.keras.Model(inp, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="cosine_similarity"
    )

    return model


# --------------------------
# Sliding windows dentro persona
# --------------------------

def sliding_inside_bbox(img, bbox, parts=3):

    x0, y0, x1, y1 = bbox
    person = img[y0:y1, x0:x1]

    h, w, _ = person.shape
    step = h // parts

    crops = []

    for i in range(parts):
        py0 = i * step
        py1 = h if i == parts - 1 else (i + 1) * step
        crop = person[py0:py1, :]
        crops.append(crop)

    return crops


# --------------------------
# ENTRENAMIENTO POR CLUSTER
# --------------------------

def train_specialists(
    product_csv,
    bundles_csv,
    match_train_csv,
    clusters_csv,
    epochs=3
):

    product_df = pd.read_csv(product_csv)
    bundles_df = pd.read_csv(bundles_csv)
    match_df = pd.read_csv(match_train_csv)
    clusters_df = pd.read_csv(clusters_csv)

    cluster_map = dict(zip(clusters_df["product_asset_id"], clusters_df["cluster"]))
    bundle_image_map = dict(zip(bundles_df["bundle_id"], bundles_df["model_image_url"]))

    segmenter = PersonSegmenter()

    base_models_dir = os.path.join(OUT_DIR, "models")
    cluster_dir = os.path.join(base_models_dir, "cluster_models")
    os.makedirs(cluster_dir, exist_ok=True)

    clusters = clusters_df["cluster"].unique()

    for cluster_id in clusters:

        print(f"\n=== TRAIN CLUSTER {cluster_id} ===")

        cluster_products = clusters_df[
            clusters_df["cluster"] == cluster_id
        ]["product_asset_id"].values

        cluster_matches = match_df[
            match_df["product_asset_id"].isin(cluster_products)
        ]

        X = []
        Y = []

        for _, row in tqdm(cluster_matches.iterrows(), total=len(cluster_matches)):

            bundle_id = row["bundle_id"]
            product_id = row["product_asset_id"]

            if bundle_id not in bundle_image_map:
                continue

            try:
                model_img = pil_to_np(download_image(bundle_image_map[bundle_id]))
                product_img = pil_to_np(
                    download_image(
                        product_df[
                            product_df["product_asset_id"] == product_id
                        ]["product_image_url"].values[0]
                    )
                )

                segmented, bbox = segmenter.segment_and_whiten(model_img)
                if bbox is None:
                    continue

                proposals = sliding_inside_bbox(segmented, bbox)

                product_img = resize_to_square(product_img, IMG_SIZE)

                for prop in proposals:

                    prop = resize_to_square(prop, IMG_SIZE)

                    X.append(prop.astype(np.float32))
                    Y.append(product_img.astype(np.float32))

            except:
                continue

        if len(X) == 0:
            print("No data for this cluster")
            continue

        X = np.array(X)
        Y = np.array(Y)

        model = build_specialist_model()

        # target = embedding producto
        target = model.predict(Y, batch_size=32, verbose=0)

        model.fit(
            X,
            target,
            epochs=epochs,
            batch_size=32
        )

        # Guardar modelo
        cluster_folder = os.path.join(cluster_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        model.save(os.path.join(cluster_folder, "specialist.keras"))

        print("Saved cluster", cluster_id)


# --------------------------
# PREDICT
# --------------------------

def predict_bundle(
    bundle_id,
    bundles_csv,
    clusters_csv,
    top_k=5
):

    bundles_df = pd.read_csv(bundles_csv)
    clusters_df = pd.read_csv(clusters_csv)

    bundle_image_map = dict(zip(bundles_df["bundle_id"], bundles_df["model_image_url"]))
    cluster_products_map = clusters_df.groupby("cluster")["product_asset_id"].apply(list).to_dict()

    segmenter = PersonSegmenter()

    img = pil_to_np(download_image(bundle_image_map[bundle_id]))
    segmented, bbox = segmenter.segment_and_whiten(img)

    if bbox is None:
        return []

    proposals = sliding_inside_bbox(segmented, bbox)

    results = []

    base_models_dir = os.path.join(OUT_DIR, "models", "cluster_models")

    for cluster_id in cluster_products_map.keys():

        model_path = os.path.join(base_models_dir, f"cluster_{cluster_id}", "specialist.keras")
        if not os.path.exists(model_path):
            continue

        model = tf.keras.models.load_model(model_path)

        cluster_products = cluster_products_map[cluster_id]

        for prop in proposals:

            prop = resize_to_square(prop, IMG_SIZE)
            emb_prop = model.predict(np.expand_dims(prop, 0), verbose=0)[0]

            for pid in cluster_products:
                results.append((pid, np.linalg.norm(emb_prop)))

    results = sorted(results, key=lambda x: x[1])

    selected = []
    used = set()

    for pid, score in results:
        if pid not in used:
            selected.append(pid)
            used.add(pid)
        if len(selected) >= top_k:
            break

    return selected


# --------------------------
# EVALUATOR
# --------------------------

def evaluate_test(
    bundles_csv,
    match_test_csv,
    clusters_csv
):

    match_df = pd.read_csv(match_test_csv)
    grouped = match_df.groupby("bundle_id")["product_asset_id"].apply(list)

    correct = 0
    total = 0

    for bundle_id, true_products in grouped.items():

        preds = predict_bundle(bundle_id, bundles_csv, clusters_csv)

        hit = any(tp in preds for tp in true_products)

        correct += int(hit)
        total += 1

    acc = correct / total
    print("Top-5 Accuracy:", acc)