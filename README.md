# Edge-Deployment-of-Compressed-CNN-Model

## Overview
This project focuses on compressing a deep convolutional neural network and deploying it for efficient inference on edge devices.  
An EfficientNetB0 model pretrained on ImageNet is converted into a fully INT8 quantized TensorFlow Lite model and served through a lightweight Flask API.

The objective is to reduce model size, improve inference speed, and make the system suitable for resource-constrained environments such as Raspberry Pi or other edge hardware.

## Features

- EfficientNetB0 pretrained on ImageNet
- Full INT8 post-training quantization using TensorFlow Lite
- REST API for predictions using Flask
- Benchmarking for latency and memory usage
- Edge-ready inference pipeline
- Clean modular code structure


Quantization converts float32 weights into int8, significantly reducing memory usage while maintaining strong predictive performance.

## Project Structure

