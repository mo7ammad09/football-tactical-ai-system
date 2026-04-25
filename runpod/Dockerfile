FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

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
