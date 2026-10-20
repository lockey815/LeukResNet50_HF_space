from fastapi import FastAPI
 
app = FastAPI()
 
@app.get("/")
def home():
    return {"message": "Hello from Python on Vercel"}
 
@app.get("/api/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import keras
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_ID = "Lockey815/LeukNet-ResNet50"
MODEL_FILENAME = "resnet50_final.keras"

def load_model():
    path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILENAME)
    return keras.saving.load_model(path, compile=False)

MODEL = load_model()

def get_target_size(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        # Channels-last: (batch, height, width, channels)
        h, w = shape[1], shape[2]
        if isinstance(h, int) and isinstance(w, int):
            return (w, h)
    except Exception:
        pass
    return (224, 224)

TARGET_SIZE = get_target_size(MODEL)

def preprocess(image, normalization):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = image.convert("RGB").resize(TARGET_SIZE)
    arr = np.asarray(image, dtype=np.float32)

    if normalization == "Scale to 0–1":
        arr = arr / 255.0
    elif normalization == "ResNet50 preprocess_input":
        arr = keras.applications.resnet50.preprocess_input(arr)
    # "Keep pixel values 0–255" intentionally performs no scaling.

    return np.expand_dims(arr, axis=0)

def predict(image, normalization, class_names):
    x = preprocess(image, normalization)
    raw = np.asarray(MODEL.predict(x, verbose=0))

    # Convert common classification output formats to a probability vector.
    values = np.squeeze(raw)
    if values.ndim == 0:
        value = float(values)
        probs = np.array([1.0 - value, value], dtype=np.float64)
    elif values.ndim == 1 and values.size == 1:
        value = float(values[0])
        probs = np.array([1.0 - value, value], dtype=np.float64)
    elif values.ndim == 1:
        values = values.astype(np.float64)
        if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1.0, atol=0.05):
            probs = values
        else:
            shifted = values - np.max(values)
            exp = np.exp(shifted)
            probs = exp / exp.sum()
    else:
        raise gr.Error(f"Unexpected model output shape: {raw.shape}. This demo expects a classification model.")

    names = [n.strip() for n in (class_names or "").split(",") if n.strip()]
    if len(names) != len(probs):
        names = [f"Class {i}" for i in range(len(probs))]

    ranked = sorted(zip(names, probs.tolist()), key=lambda item: item[1], reverse=True)
    label, confidence = ranked[0]
    chart = {name: float(prob) for name, prob in ranked}
    note = (
        f"**Predicted class:** {label}\n\n"
        f"**Model confidence:** {confidence:.1%}\n\n"
        "Research/demo output only. This result is not a medical diagnosis."
    )
    return note, chart

with gr.Blocks(theme=gr.themes.Soft(), title="LeukNet-ResNet50 Demo") as demo:
    gr.Markdown(
        """
        # LeukNet-ResNet50
        Upload a microscopy image to run the image-classification model hosted on Hugging Face.

        **Important:** Select the same preprocessing and class order used during training. The model repository does not document those settings, so verify them before interpreting predictions.
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
                info="Replace these with the exact labels and order used when training the model.",
            )
            run_button = gr.Button("Classify image", variant="primary")
        with gr.Column():
            result = gr.Markdown(label="Prediction")
            probabilities = gr.Label(label="Class scores", num_top_classes=10)
    run_button.click(
        fn=predict,
        inputs=[image_input, normalization, class_names],
        outputs=[result, probabilities],
    )
    gr.Markdown(
        f"Model repository: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · "
        "This demo does not replace review by a qualified laboratory professional."
    )

if __name__ == "__main__":
    demo.launch()
