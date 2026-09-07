FROM python:3.11-slim

# Install system dependencies (mupdf / poppler / build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, data, and seed database
COPY fact_layer/ ./fact_layer/
COPY starter-datasets/ ./starter-datasets/
COPY scripts/ ./scripts/
COPY pytest.ini .
COPY tests/ ./tests/
COPY data/seed_data.sqlite3 ./data/seed_data.sqlite3

# Pre-create data directories, pre-seed SQLite database, and copy PDFs
RUN mkdir -p data/pages data/uploads data/pdfs && \
    cp ./data/seed_data.sqlite3 ./data/fact_layer.sqlite3 && \
    find starter-datasets/ -name "*.pdf" -exec cp {} data/pdfs/ \;

# Expose port
EXPOSE 8000

# Default environment
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Start command
CMD ["sh", "-c", "uvicorn fact_layer.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
