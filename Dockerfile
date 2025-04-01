FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Erstelle die notwendigen Verzeichnisse
RUN mkdir -p /app/dashboard/templates /app/dashboard/static

# Kopiere die Anwendungscode
COPY . .

# Setze die Berechtigungen
RUN chmod -R 755 /app

EXPOSE 5001

ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]