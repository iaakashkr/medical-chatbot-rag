# Stage 1: Build stage
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build tools only in this stage
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.10-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the whole project folder structure
COPY . .

EXPOSE 8000

ENV RUN_MODE="fastapi"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
