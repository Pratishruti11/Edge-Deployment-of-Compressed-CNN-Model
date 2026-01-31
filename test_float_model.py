import tensorflow as tf
import numpy as np
from PIL import Image

# Load labels
with open("imagenet_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()[1:]]

print("Loading FLOAT EfficientNet...")

model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=True
)

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img).astype(np.float32)

    # VERY IMPORTANT
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


image_path = "dog.png"  

input_data = preprocess(image_path)

print("Running FLOAT inference...")

preds = model.predict(input_data)

top5 = preds[0].argsort()[-5:][::-1]

print("\nTOP PREDICTIONS:\n")

for i in top5:
    print(
        f"{i+1} | {labels[i]} | confidence={preds[0][i]:.3f}"
    )
