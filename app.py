"""Streamlit demo: real vs AI-generated image classifier.

Run locally with:
    streamlit run app.py

Deployed on HuggingFace Spaces — model weights are pulled from
HF Hub on first use and cached to disk. See README.md for details.
"""

import streamlit as st
from PIL import Image

from src.predict import MODEL_REGISTRY, load_model, predict_image


st.set_page_config(
    page_title="Real vs AI-generated image classifier",
    page_icon="\U0001F5BC️",
    layout="centered",
)

st.title("Real vs AI-generated image classifier")
st.write(
    "Course project for DAT255 — Deep Learning Engineering. "
    "Pick a model, upload an image, and see whether the model thinks "
    "it's a real photograph or AI-generated."
)


@st.cache_resource(show_spinner="Loading model weights...")
def _get_model(tag: str):
    return load_model(tag, device="cpu")


tag_by_label = {spec.display_name: tag for tag, spec in MODEL_REGISTRY.items()}

chosen_label = st.selectbox(
    "Model",
    list(tag_by_label.keys()),
    index=0,
    help="Test AUC on the held-out test set is shown in the caption below.",
)
chosen_tag = tag_by_label[chosen_label]
chosen_spec = MODEL_REGISTRY[chosen_tag]
st.caption(f"Test AUC: {chosen_spec.test_auc:.4f}")

uploaded = st.file_uploader(
    "Upload an image (JPG, PNG, WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Your image", use_column_width=True)

    model = _get_model(chosen_tag)
    with st.spinner("Running inference..."):
        prob_ai, label = predict_image(model, image, device="cpu")

    if label == "AI-generated":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    st.write(f"Probability the image is AI-generated: **{prob_ai:.2%}**")
    st.progress(prob_ai)

    with st.expander("What does this number mean?"):
        st.write(
            "The model outputs a single number between 0 and 1 "
            "(a sigmoid of its internal logit). 0 means confidently real, "
            "1 means confidently AI-generated. The label above uses a "
            "threshold of 0.5."
        )

st.divider()
st.caption(
    "Models were trained on a 60 000-image dataset split 80/10/10. "
    "The three transfer-learning models fine-tune ImageNet-pretrained "
    "backbones; the scratch ResNet-50 was trained from random "
    "initialisation with ReLU replaced by GELU throughout the network."
)
