"""Single-image inference for the Streamlit demo.

Loads a trained checkpoint (local or from HuggingFace Hub) and returns a
probability that an uploaded image is AI-generated.

Label convention follows the training pipeline in src/dataset.py:
    0 = real, 1 = AI-generated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from src.models import create_model
from src.transforms import get_eval_transforms


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    display_name: str
    model_name: str
    modified: bool
    test_auc: float


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "vit_base_patch16_224_transfer": ModelSpec(
        tag="vit_base_patch16_224_transfer",
        display_name="ViT-B/16 (transfer learning) — best accuracy",
        model_name="vit_base_patch16_224",
        modified=False,
        test_auc=0.9950,
    ),
    "densenet121_transfer": ModelSpec(
        tag="densenet121_transfer",
        display_name="DenseNet-121 (transfer learning)",
        model_name="densenet121",
        modified=False,
        test_auc=0.9854,
    ),
    "resnet50_transfer": ModelSpec(
        tag="resnet50_transfer",
        display_name="ResNet-50 (transfer learning)",
        model_name="resnet50",
        modified=False,
        test_auc=0.9749,
    ),
    "resnet50_scratch_gelu": ModelSpec(
        tag="resnet50_scratch_gelu",
        display_name="ResNet-50 (from scratch, GELU activation)",
        model_name="resnet50",
        modified=True,
        test_auc=0.9349,
    ),
}

WEIGHTS_REPO_DEFAULT = "Joergenator/imagerecogniser-weights"

_EVAL_TRANSFORM = get_eval_transforms()


def _resolve_checkpoint(tag: str) -> str:
    local = os.path.join("results", "checkpoints", tag, "best.pt")
    if os.path.isfile(local):
        return local
    repo_id = os.environ.get("HF_WEIGHTS_REPO", WEIGHTS_REPO_DEFAULT)
    return hf_hub_download(repo_id=repo_id, filename=f"{tag}.pt")


def load_model(tag: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    if tag not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model tag: {tag}. Known tags: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[tag]
    ckpt_path = _resolve_checkpoint(tag)

    model = create_model(
        model_name=spec.model_name,
        pretrained=False,
        dropout=0.3,
        modified=spec.modified,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: str | torch.device = "cpu",
) -> tuple[float, str]:
    img = image.convert("RGB")
    tensor = _EVAL_TRANSFORM(img).unsqueeze(0).to(device)
    logit = model(tensor).squeeze()
    prob = torch.sigmoid(logit).item()
    label = "AI-generated" if prob >= 0.5 else "Real"
    return prob, label
