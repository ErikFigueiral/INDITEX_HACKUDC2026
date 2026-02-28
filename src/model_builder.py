import tensorflow as tf

IMG_SIZE = 224


def build_embedding_model(trainable=False):

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # 🔥 Congelamos backbone
    base.trainable = trainable

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # 🔥 SIN training=False
    x = base(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1)
    )(x)

    model = tf.keras.Model(inputs, x)

    return model