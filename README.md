---
title: Real vs AI-generated image classifier
emoji: "\U0001F5BC️"
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
---

# Real vs AI-generated image classifier

Course project for **DAT255 — Deep Learning Engineering**. Given an uploaded image, four different models predict whether it's a real photograph or AI-generated.

## Try it

Pick a model from the dropdown, upload a JPG/PNG/WebP, and the app returns the probability that the image is AI-generated.

## Models

| Model | Strategy | Test AUC |
|---|---|---|
| ViT-B/16 (transfer learning) | Fine-tuned ImageNet backbone | 0.9950 |
| DenseNet-121 (transfer learning) | Fine-tuned ImageNet backbone | 0.9854 |
| ResNet-50 (transfer learning) | Fine-tuned ImageNet backbone | 0.9749 |
| ResNet-50 (from scratch, GELU) | Random init, ReLU → GELU | 0.9349 |

All four were trained on a 60 000-image dataset split 80/10/10. Weights are hosted on the HuggingFace Hub and pulled at first use.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app prefers local weights at `results/checkpoints/<tag>/best.pt` if they exist; otherwise it downloads from the configured HF Hub repo. Override the repo with `HF_WEIGHTS_REPO=username/reponame`.

## Source

Training code and experiment history: [github.com/Joergenator/ImageRecogniser](https://github.com/Joergenator/ImageRecogniser). The four `scratch-resnet50-gelu*` branches record the iterations behind the from-scratch model.
