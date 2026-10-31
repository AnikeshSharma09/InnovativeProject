# ==============================================================================
# Production Dockerfile for Real-Time ASL Sign Language AI & Translation Engine
# ==============================================================================

FROM python:3.11-slim

# System metadata
LABEL maintainer="Anikesh Sharma <anikesh09>"
LABEL description="Real-Time Motion-Aware ASL Recognition & Linguistic Translation Engine"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=3 \
    GLOG_minloglevel=3 \
    PORT=8501

# Install required system libraries (OpenCV, MediaPipe, Speech TTS, FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    libespeak1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install lightweight CPU-optimized PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache MediaPipe Holistic and Hands models inside the container image
# Ensures 100% offline execution without runtime downloads
RUN python -c "import mediapipe as mp; mp.solutions.holistic.Holistic(model_complexity=0); mp.solutions.hands.Hands(max_num_hands=2); print('[+] MediaPipe perception models pre-cached successfully')"

# Copy project files into container
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default execution command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
