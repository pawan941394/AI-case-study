FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (needed for sklearn/numpy builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps
# Copy project files
COPY . .

# Create folders for SQLite + embeddings
RUN mkdir -p /app/tmp
RUN mkdir -p /app/tmp/embeddings

EXPOSE 8000

CMD ["uvicorn", "api_For_bot:app", "--host", "0.0.0.0", "--port", "8000"]
