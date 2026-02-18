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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into system site-packages (/usr/local)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt


# -------------------------
# Runtime stage
# -------------------------
FROM python:3.11-slim

WORKDIR /app

# Install runtime OS dependencies (PDF + OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python deps from builder (system-wide)
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY app.py .
COPY docint/ ./docint/
COPY data_model/ ./data_model/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Healthcheck: keep it simple and secret-free (use your /health as-is)
# If /health is protected, strongly consider making it unauthenticated on localhost,
# or do healthchecks at Traefik instead.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Run the application (module form avoids PATH/script issues)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
