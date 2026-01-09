# =========================
# Base Image
# =========================
FROM python:3.11-slim

# =========================
# Environment
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# Workdir
# =========================
WORKDIR /app

# =========================
# System deps (خفيف + ضروري)
# =========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Install Python deps
# =========================
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# =========================
# Copy Project
# =========================
COPY app ./app

# =========================
# Expose Port
# =========================
EXPOSE 8000

# =========================
# Start FastAPI (Railway uses PORT)
# =========================
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
