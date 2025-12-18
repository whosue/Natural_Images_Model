# Natural Image Classification using Transfer Learning

## Overview

This project implements a deep learning image classification model using transfer learning.

## Dataset

The Natural Images dataset was obtained from Kaggle and contains 6,899 images across the following classes:

- airplane
- car
- cat
- dog
- flower
- fruit
- motorbike
- person

The dataset is split into 80% training and 20% validation data.

## Model

- Architecture: MobileNetV2 (pre-trained on ImageNet)
- Framework: TensorFlow / Keras
- Loss: Sparse Categorical Cross-Entropy
- Optimizer: Adam
- Input size: 224×224

## How to Run

```bash
pip install -r requirements.txt
python train.py
```
