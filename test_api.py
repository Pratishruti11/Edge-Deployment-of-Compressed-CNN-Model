# test_api.py
import requests
import sys

image_file = sys.argv[1] if len(sys.argv) > 1 else 'image.jpg'

with open(image_file, 'rb') as f:
    r = requests.post('http://localhost:5000/predict', files={'image': f})
    print(f"Testing: {image_file}")
    print(r.json())