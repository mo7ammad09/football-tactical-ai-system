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
RUN pip install --no-cache-dir --no-deps boxmot==12.0.2

# StrongSORT/OSNet ReID weights are baked into the image so cold starts do not
# spend time downloading them during the first analysis job.
ENV STRONGSORT_REID_WEIGHTS=/app/tracking/weights/osnet_ain_x1_0_msmt17.pt
RUN mkdir -p /app/tracking/weights && python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth', '/app/tracking/weights/osnet_ain_x1_0_msmt17.pt')"

# Copy the official source tree only.
COPY src/ ./src/
COPY runpod/handler.py .
RUN mkdir -p models

CMD ["python", "handler.py"]
