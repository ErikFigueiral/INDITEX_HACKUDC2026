---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2426837728.py in <cell line: 0>()
      4 df = load_dataset("product_dataset")
      5 
----> 6 generate_embeddings(
      7     df,
      8     id_col="product_asset_id",

/kaggle/working/INDITEX_HACKUDC2026/src/embedder.py in generate_embeddings(df, id_col, url_col, limit)
     11     "User-Agent": (
     12         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
---> 13         "AppleWebKit/537.36 (KHTML, like Gecko) "
     14         "Chrome/121.0 Safari/537.36"
     15     )

/kaggle/working/INDITEX_HACKUDC2026/src/model_builder.py in build_embedding_model(trainable)
     17     inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
     18 
---> 19     x = base(inputs)
     20     x = layers.GlobalAveragePooling2D()(x)
     21 

/usr/local/lib/python3.12/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.12/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    154 
    155     def __tf_tensor__(self, dtype=None, name=None):
--> 156         raise ValueError(
    157             "A KerasTensor cannot be used as input to a TensorFlow function. "
    158             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.ops`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```