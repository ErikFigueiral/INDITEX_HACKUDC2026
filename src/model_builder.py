import keras
from keras import layers

IMG_SIZE = 224


def build_embedding_model(trainable=False):

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base.trainable = trainable

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)

    # L2 normalization compatible Keras 3
    x = layers.Lambda(lambda t: keras.ops.normalize(t, axis=1))(x)

    model = keras.Model(inputs, x)

    return model