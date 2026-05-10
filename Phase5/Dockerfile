FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Phase1/ ./phase1/
COPY Phase2/ ./phase2/
COPY Phase3/ ./phase3/
COPY Phase4/ ./phase4/

RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    soundfile==0.12.1 \
    scikit-learn==1.4.0 \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    websockets==12.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.4

RUN pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH=/app/phase1:/app/phase2:/app/phase3:/app/phase4

EXPOSE 8765
EXPOSE 8000

CMD ["python", "phase4/main.py", "--mode", "web", "--config", "phase4/config.yaml"]