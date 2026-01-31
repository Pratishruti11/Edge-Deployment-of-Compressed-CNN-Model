import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading pretrained EfficientNetB0...")

model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=True,
    input_shape=(224,224,3)
)

model.export(f"{MODEL_DIR}/saved_model")

print("Creating representative dataset...")

def representative_dataset():
    for _ in range(100):
        img = np.random.randint(
            0,256,(1,224,224,3)
        ).astype(np.float32)

        yield [img]


print("Converting to FULL INT8...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    f"{MODEL_DIR}/saved_model"
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open(f"{MODEL_DIR}/model_int8.tflite","wb") as f:
    f.write(tflite_model)

print(f"\nSUCCESS — INT8 model size: {len(tflite_model)/1e6:.2f} MB")