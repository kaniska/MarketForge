FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Download the trained model at build time so it's baked into the image.
# Falls back to the base (untrained) model at runtime if this path is empty.
# To use a fine-tuned checkpoint, set MODEL_REPO to your HF model repo.
ENV MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct"
ENV MODEL_DIR="/app/trained-model"
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
repo = os.environ.get('MODEL_REPO', 'Qwen/Qwen2.5-0.5B-Instruct'); \
snapshot_download(repo_id=repo, local_dir=os.environ['MODEL_DIR']); \
print(f'Downloaded {repo} to {os.environ[\"MODEL_DIR\"]}')" \
    || echo "Model download skipped (will download at runtime)"

EXPOSE 7860 8000

ENV ENABLE_WEB_INTERFACE=true
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Start both the API server and the visual dashboard
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000 & python app_visual.py"]
