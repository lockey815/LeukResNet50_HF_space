FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

# Bake the model into the image so cold starts don't re-download it.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Lockey815/LeukNet-ResNet50', 'resnet50_final.keras')" \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "app.py"]
