from PIL import Image
import os

if os.path.exists('image.jpg'):
    img = Image.open('image.jpg')
    print(f"Image found")
    print(f"Size: {img.size}")
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")

    img.show()   

else:
    print("Image not found")
