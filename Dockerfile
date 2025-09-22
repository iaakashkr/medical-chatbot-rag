# Base Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole repo
COPY . .

# Optional: expose a port for future FastAPI usage
EXPOSE 8000

# Environment variables for Gemini API keys (override at runtime)
ENV GEMINI_API_KEY_1=""
ENV GEMINI_MODEL="gemini-1.5-flash"

# Default command: run your CLI chatbot
CMD ["python", "-m", "app.main"]
