FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Trusted Ultralytics checkpoints saved with older torch/YOLO versions need
# classic torch.load behavior under PyTorch 2.6+.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# System dependencies for OpenCV/video handling.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install worker requirements.
COPY runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the official source tree only.
COPY src/ ./src/
COPY runpod/handler.py .
RUN mkdir -p models

CMD ["python", "handler.py"]
