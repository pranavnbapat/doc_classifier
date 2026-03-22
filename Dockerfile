# syntax=docker/dockerfile:1

# -------------------------
# Build stage
# -------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (only needed for building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into system site-packages (/usr/local)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt


# -------------------------
# Runtime stage
# -------------------------
FROM python:3.11-slim

WORKDIR /app

ARG AGRI_EMBEDDING_MODEL=intfloat/multilingual-e5-small

# Install runtime OS dependencies (PDF + OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

ENV AGRI_EMBEDDING_MODEL=${AGRI_EMBEDDING_MODEL}
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Copy installed Python deps from builder (system-wide)
COPY --from=builder /usr/local /usr/local

# Pre-download the agriculture embedding model so first inference is warm.
RUN mkdir -p /app/.cache/huggingface \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${AGRI_EMBEDDING_MODEL}', device='cpu')"

# Copy application code
COPY app.py .
COPY docint/ ./docint/
COPY data_model/ ./data_model/

# Create non-root user for security
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck: /health is intentionally unauthenticated for local/container probes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Run the application (module form avoids PATH/script issues)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
