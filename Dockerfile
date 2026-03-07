FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    fastapi \
    "uvicorn[standard]" \
    gradio \
    matplotlib \
    numpy \
    requests

EXPOSE 7860 8000

ENV ENABLE_WEB_INTERFACE=true
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Start both the API server and the visual dashboard
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000 & python app_visual.py"]
