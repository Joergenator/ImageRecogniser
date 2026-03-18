# Differentiating Between AI-Generated and Human-Made Images

## Objective

Develop a deep learning model for binary classification: **AI-generated** vs. **human-made** images. Primary metric: **AUC (area under the ROC curve)**. Secondary metrics: accuracy, precision, recall, F1-score.

---

## 1. Dataset

**Source:** [AI-Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images) (~60,000 images, ~30k per class)

### 1.1 Download and Organization

1. Download the dataset via the Kaggle API:
   ```bash
   kaggle datasets download -d tristanzhang32/ai-generated-images-vs-real-images
   unzip ai-generated-images-vs-real-images.zip -d data/
   ```
2. Verify the folder structure:
   ```
   data/
   ├── train/
   │   ├── ai/
   │   └── real/
   ├── val/          # if provided, otherwise create manually
   └── test/
   ```

### 1.2 Data Split

If the dataset does not include a predefined validation/test split, create one:

| Split      | Proportion | Purpose                          |
|------------|------------|----------------------------------|
| Train      | 70%        | Model training                   |
| Validation | 15%        | Hyperparameter tuning, early stopping |
| Test       | 15%        | Final evaluation (report results)|

Use stratified splitting to maintain class balance across splits.

### 1.3 Preprocessing

- Resize images to **224×224** (ResNet/DenseNet) or **224×224 / 384×384** (ViT, depending on variant).
- Normalize with ImageNet mean and std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- Basic augmentation for the baseline: random horizontal flip, random crop.

---

## 2. Models

### 2.1 ResNet (e.g., ResNet-50)

- Well-established CNN with residual connections.
- Available pretrained on ImageNet via `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`.
- Replace the final `fc` layer with `nn.Linear(2048, 1)` for binary classification (use `BCEWithLogitsLoss`).

### 2.2 DenseNet (e.g., DenseNet-121)

- Dense connections encourage feature reuse, often more parameter-efficient than ResNet.
- Replace the `classifier` layer with `nn.Linear(1024, 1)`.

### 2.3 Vision Transformer (ViT)

- Transformer-based architecture; processes images as patch sequences.
- Use `timm` library: `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)`.
- Typically requires more data or strong pretraining to outperform CNNs.

---

## 3. Training Strategies

### 3.1 Strategy A — Transfer Learning with Fine-Tuning

**Phase 1: Train the classification head**
1. Load pretrained ImageNet weights.
2. Freeze all backbone layers.
3. Train only the new classification head for 5–10 epochs.
4. Use a moderate learning rate (e.g., `1e-3`).

**Phase 2: Fine-tune the full network**
1. Unfreeze all layers.
2. Use a much lower learning rate (e.g., `1e-5` to `5e-5`) — optionally use discriminative learning rates (lower LR for early layers, higher for later layers).
3. Train for 10–30 epochs with early stopping (patience ~5 epochs, monitor validation AUC).

```python
# Pseudocode
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(2048, 1)

# Phase 1: freeze backbone
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
train(model, lr=1e-3, epochs=5)

# Phase 2: unfreeze all
model.requires_grad_(True)
train(model, lr=1e-5, epochs=20)
```

### 3.2 Strategy B — Training from Scratch

1. Initialize all weights randomly (no pretrained weights).
2. Use a higher learning rate (e.g., `1e-2` with warm-up).
3. Train for more epochs (50–100+), since the network must learn low-level features from scratch.
4. Expect lower performance unless the dataset is very large or highly specialized.

This serves as a **baseline** to quantify the benefit of transfer learning.

### 3.3 Strategy C — Advanced Augmentation and Regularization

Apply on top of Strategy A (transfer learning) to push performance further:

| Technique               | Details                                                    |
|--------------------------|------------------------------------------------------------|
| **MixUp**               | Blend pairs of images and labels: `x = λ·x_i + (1-λ)·x_j` |
| **CutMix**              | Replace image patches with patches from another image       |
| **Weight Decay**        | L2 regularization, typically `1e-4` to `5e-4`              |
| **Dropout**             | Add dropout before the classification head (e.g., `p=0.3`) |
| **Label Smoothing**     | Soften targets (e.g., `0.1`)                                |
| **Cosine Annealing LR** | `CosineAnnealingLR` or `CosineAnnealingWarmRestarts`       |
| **Stochastic Depth**    | Randomly drop layers during training (built into `timm`)    |

```python
# MixUp example
lam = np.random.beta(0.4, 0.4)
index = torch.randperm(x.size(0))
mixed_x = lam * x + (1 - lam) * x[index]
loss = lam * criterion(pred, y) + (1 - lam) * criterion(pred, y[index])
```

---

## 4. Training Infrastructure

### 4.1 Framework and Libraries

```
torch >= 2.0
torchvision
timm                # for ViT and advanced models
scikit-learn        # for metrics (roc_auc_score, classification_report)
matplotlib / seaborn # for plots
wandb or tensorboard # for experiment tracking
```

### 4.2 Hardware

- Use GPU (CUDA). Training on CPU is impractical for this dataset size.
- Kaggle notebooks, Google Colab (free T4), or local GPU.
- Mixed precision training (`torch.amp`) to speed up training and reduce memory.

### 4.3 Reproducibility

- Set random seeds: `torch.manual_seed(42)`, `np.random.seed(42)`.
- Use `torch.backends.cudnn.deterministic = True`.
- Log all hyperparameters per experiment.

---

## 5. Evaluation

### 5.1 Metrics

Compute on the **test set** (never used during training or tuning):

| Metric     | How to compute                              |
|------------|---------------------------------------------|
| **AUC**    | `sklearn.metrics.roc_auc_score(y, probs)`   |
| Accuracy   | `sklearn.metrics.accuracy_score(y, preds)`  |
| Precision  | `sklearn.metrics.precision_score(y, preds)` |
| Recall     | `sklearn.metrics.recall_score(y, preds)`    |
| F1-score   | `sklearn.metrics.f1_score(y, preds)`        |

### 5.2 Visualizations

- **ROC Curve**: plot for each model/strategy on the same axes for comparison.
- **Confusion Matrix**: per model.
- **Training curves**: loss and AUC over epochs (train vs. validation).
- **GradCAM / attention maps**: visualize what the model focuses on (useful for the report — shows whether the model learned meaningful features).

### 5.3 Comparison Table

| Model      | Strategy              | AUC  | Accuracy | F1   | Notes          |
|------------|-----------------------|------|----------|------|----------------|
| ResNet-50  | A (Transfer Learning) |      |          |      |                |
| ResNet-50  | B (From Scratch)      |      |          |      |                |
| ResNet-50  | C (Adv. Augmentation) |      |          |      |                |
| DenseNet   | A (Transfer Learning) |      |          |      |                |
| DenseNet   | B (From Scratch)      |      |          |      |                |
| DenseNet   | C (Adv. Augmentation) |      |          |      |                |
| ViT        | A (Transfer Learning) |      |          |      |                |
| ViT        | B (From Scratch)      |      |          |      |                |
| ViT        | C (Adv. Augmentation) |      |          |      |                |

---

## 6. Work Distribution (4 Members)

| Member | Responsibility                                              |
|--------|-------------------------------------------------------------|
| 1      | Data pipeline (download, split, preprocessing, dataloaders) |
| 2      | Strategy A — Transfer learning with fine-tuning (all models)|
| 3      | Strategy B — Training from scratch (all models)             |
| 4      | Strategy C — Advanced augmentation/regularization + evaluation/plots |

All members contribute to the final report and analysis. Use a shared experiment tracker (e.g., Weights & Biases) so results are centralized.

---

## 7. Suggested Project Structure

```
ImageRecogniser/
├── data/                    # dataset (gitignored)
├── src/
│   ├── dataset.py           # Dataset class, transforms, dataloaders
│   ├── models.py            # Model definitions (ResNet, DenseNet, ViT)
│   ├── train.py             # Training loop, loss, optimizer setup
│   ├── evaluate.py          # Metrics computation, plotting
│   └── utils.py             # Seeds, logging, misc helpers
├── notebooks/               # exploratory analysis, visualization
├── configs/                 # hyperparameter configs (YAML/JSON)
├── results/                 # saved metrics, plots, model checkpoints
├── requirements.txt
├── project_plan.md
└── README.md
```

---

## 8. Key Risks and Mitigations

| Risk                                  | Mitigation                                             |
|---------------------------------------|--------------------------------------------------------|
| Overfitting (especially from scratch) | Early stopping, dropout, weight decay, augmentation    |
| Class imbalance                       | Verify balance; use stratified splits; monitor per-class metrics |
| ViT underperforming with limited data | Use strong pretrained weights; consider smaller ViT variants |
| Long training times                   | Mixed precision, smaller batch sizes, gradient accumulation |
| Irreproducible results                | Fix seeds, log all hyperparameters, version control    |
