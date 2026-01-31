import tensorflow as tf
import numpy as np
import time
import psutil
import os

MODEL_PATH = "model/model_int8.tflite"

# Model size
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dummy_input = np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)

# Warmup
for _ in range(5):
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

# Latency measurement
runs = 100
start = time.time()
for _ in range(runs):
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
end = time.time()

latency_ms = (end - start) / runs * 1000

# Memory usage
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / (1024 ** 2)

# Results
print("="*50)
print("BENCHMARK RESULTS")
print("="*50)
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Avg Inference Latency: {latency_ms:.2f} ms")
print(f"Memory Usage: {memory_mb:.2f} MB")
print(f"Throughput: {1000/latency_ms:.1f} images/sec")
print("="*50)

# Comparison (for report)
original_efficientnet_size = 16.5  # MB (approx)
compression_ratio = original_efficientnet_size / model_size_mb
print(f"\nCompression: {compression_ratio:.1f}x smaller than original")
