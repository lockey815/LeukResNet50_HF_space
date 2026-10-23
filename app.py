import os

# Configure Keras before importing it.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import keras
from huggingface_hub import hf_hub_download

MODEL_ID = "Lockey815/LeukNet-ResNet50"
MODEL_FILENAME = "resnet50_final.keras"


def load_classifier():
    """Download the model artifact from Hugging Face and load it with Keras."""
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    return keras.saving.load_model(model_path, compile=False)


MODEL = load_classifier()


def get_target_size(model):
    """Return PIL's (width, height) from a channels-last model input shape."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) >= 4:
            height, width = shape[1], shape[2]
            if isinstance(height, int) and isinstance(width, int):
                return (width, height)
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    return (224, 224)


TARGET_SIZE = get_target_size(MODEL)


def preprocess(image, normalization):
    if image is None:
        raise gr.Error("Upload an image first.")

    image = image.convert("RGB").resize(TARGET_SIZE)
    array = np.asarray(image, dtype=np.float32)

    if normalization == "Scale to 0–1":
        array = array / 255.0
    elif normalization == "ResNet50 preprocess_input":
        array = keras.applications.resnet50.preprocess_input(array)
    # "Keep pixel values 0–255" intentionally leaves the values unchanged.

    return np.expand_dims(array, axis=0)


def output_to_probabilities(raw):
    """Handle common sigmoid, softmax, and logits classification outputs."""
    values = np.asarray(raw, dtype=np.float64).squeeze()

    if values.ndim == 0:
        value = float(np.clip(values, 0.0, 1.0))
        return np.array([1.0 - value, value], dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise gr.Error(
            f"Unexpected model output shape: {np.asarray(raw).shape}. "
            "This demo expects a classification model."
        )

    if values.size == 1:
        value = float(np.clip(values[0], 0.0, 1.0))
        return np.array([1.0 - value, value], dtype=np.float64)

    if (
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(values.sum(), 1.0, atol=0.05)
    ):
        return values / values.sum()

    if not np.all(np.isfinite(values)):
        raise gr.Error("The model returned non-finite values.")

    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def predict(image, normalization, class_names):
    x = preprocess(image, normalization)
    raw = MODEL.predict(x, verbose=0)
    probabilities = output_to_probabilities(raw)

    names = [name.strip() for name in (class_names or "").split(",") if name.strip()]
    if len(names) != len(probabilities):
        names = [f"Class {index}" for index in range(len(probabilities))]

    ranked = sorted(
        zip(names, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    label, confidence = ranked[0]
    chart = {name: float(score) for name, score in ranked}
    note = (
        f"**Predicted class:** {label}\n\n"
        f"**Model score:** {confidence:.1%}\n\n"
        "Research/demo output only. This result is not a medical diagnosis."
    )
    return note, chart


with gr.Blocks(theme=gr.themes.Soft(), title="LeukNet-ResNet50 Demo") as demo:
    gr.Markdown(
        f"""
# LeukNet-ResNet50

Upload a microscopy image to run the image-classification model hosted on
[Hugging Face](https://huggingface.co/{MODEL_ID}).

**Important:** Choose the same preprocessing method and class order used during
training. Verify these settings before interpreting predictions.
"""
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload microscopy image")
            normalization = gr.Radio(
                choices=[
                    "Scale to 0–1",
                    "ResNet50 preprocess_input",
                    "Keep pixel values 0–255",
                ],
                value="Scale to 0–1",
                label="Image preprocessing",
            )
            class_names = gr.Textbox(
                value="Class 0, Class 1",
                label="Class names in model output order (comma-separated)",
                info="Enter the exact labels and order used during model training.",
            )
            run_button = gr.Button("Classify image", variant="primary")

        with gr.Column():
            result = gr.Markdown()
            probabilities = gr.Label(label="Class scores", num_top_classes=10)

    run_button.click(
        fn=predict,
        inputs=[image_input, normalization, class_names],
        outputs=[result, probabilities],
    )

    gr.Markdown(
        "This demo is for research and education. It does not replace review by "
        "a qualified laboratory professional."
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
