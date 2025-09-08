# Newer PyTorch with CUDA 12.4
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install torch/vision that match cu124
ARG TORCH=2.4.0
ARG TV=0.19.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==${TORCH} torchvision==${TV}

# Now the rest of your deps (DO NOT pin torch/torchvision again here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# *** This was missing ***
COPY . .

ENV PYTHONPATH=/app \
    WEIGHTS_PATH=/app/app/weights/model.pt \
    THRESHOLD=0.50 \
    VIDEO_FPS=1

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
