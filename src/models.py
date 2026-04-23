"""Model factory for binary classification: ResNet-50, DenseNet-121, ViT-B/16."""

import timm
import torch.nn as nn


SUPPORTED_MODELS = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "vit_base_patch16_224": "vit_base_patch16_224",
}


def _replace_relu_with_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            _replace_relu_with_gelu(child)


def create_model(model_name="resnet50", pretrained=True, dropout=0.3, modified=False):
    """Create a binary classification model.

    Args:
        model_name: One of 'resnet50', 'densenet121', 'vit_base_patch16_224'.
        pretrained: Use ImageNet-pretrained weights.
        dropout: Dropout rate before the final classifier.
        modified: If True, replace ReLU with GELU in ResNet-50.

    Returns:
        model: nn.Module with a single-output (sigmoid) head.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(SUPPORTED_MODELS)}")

    model = timm.create_model(
        SUPPORTED_MODELS[model_name],
        pretrained=pretrained,
        num_classes=1,
        drop_rate=dropout,
    )

    if modified and model_name == "resnet50":
        _replace_relu_with_gelu(model)

    return model


def freeze_backbone(model):
    """Freeze all parameters except the classification head."""
    classifier_params = set(id(p) for p in model.get_classifier().parameters())
    for param in model.parameters():
        if id(param) not in classifier_params:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
