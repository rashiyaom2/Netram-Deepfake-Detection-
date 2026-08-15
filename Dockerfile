# Netram AI Deepfake Detection Backend Dockerfile
# Optimized for Railway, Render, and Production Container Deployments

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8765

WORKDIR /app

# Install system dependencies required by OpenCV, MediaPipe, Librosa, Soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first (fast download ~150MB instead of 4GB CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all codebase, weights, models, and pipeline
COPY . .

# Expose port (Railway will override PORT at runtime)
EXPOSE 8765

# Start the application
CMD ["python", "main.py"]
