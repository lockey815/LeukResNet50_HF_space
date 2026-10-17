---
title: LeukNet-ResNet50 Demo
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

# LeukNet-ResNet50 Demo

A Gradio demo for the Keras model at
[Lockey815/LeukNet-ResNet50](https://huggingface.co/Lockey815/LeukNet-ResNet50).

The Space downloads `resnet50_final.keras` from the model repository at startup.

## Before interpreting results

The model repository does not document the exact class names, output order, or image preprocessing. In the app, set the class names in the same order used during training and choose the preprocessing that matches the training pipeline. The default is scaling pixels to `[0, 1]`, which may need to be changed.

This is an educational/research demonstration, not a diagnostic medical device.
