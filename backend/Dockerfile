# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code including models
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start command with simple gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300"]
