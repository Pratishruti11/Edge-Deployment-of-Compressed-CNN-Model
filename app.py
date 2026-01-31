
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

with open("imagenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()[1:]]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image):

    image = image.resize((224,224))
    image = np.array(image).astype(np.float32)

    # EfficientNet normalization
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    # GET quantization params
    scale, zero_point = input_details[0]["quantization"]

    # convert float → uint8
    image = image / scale + zero_point
    image = np.clip(image, 0, 255).astype(np.uint8)

    image = np.expand_dims(image, axis=0)

    return image



@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = Image.open(request.files["image"]).convert("RGB")
        input_data = preprocess(image)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])[0]

        scale, zero_point = output_details[0]["quantization"]

        logits = scale * (output - zero_point)

        
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])


        return jsonify({
            "class_id": pred_class+1,
            "class_name": labels[pred_class],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print("SERVER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

