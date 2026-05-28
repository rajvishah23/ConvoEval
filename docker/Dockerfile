FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Pre-process facets on build if CSV exists
RUN if [ -f data/facets.csv ]; then python scripts/preprocess_facets.py --input data/facets.csv --output data/facets_processed.csv; fi

EXPOSE 8000

ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=qwen2.5:3b
ENV FACETS_PATH=data/facets_processed.csv

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
