FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies and Python packages with higher timeout/retries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && pip install --no-cache-dir --timeout=600 --retries=5 --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --timeout=600 --retries=5 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2+cpu torchaudio==2.2.2+cpu \
    && pip install --no-cache-dir --timeout=600 --retries=5 -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/agent/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
