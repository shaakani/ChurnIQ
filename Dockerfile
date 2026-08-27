# ChurnIQ model API - container image
#
# Build:  docker build -t churniq-api .
# Run:    docker run -p 8000:8000 churniq-api
# Then:   curl http://127.0.0.1:8000/health

FROM python:3.11-slim

WORKDIR /app

# Install only the serving dependencies (better layer caching on rebuilds,
# and a smaller image than installing the full training requirements.txt)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs to run - not the raw data generation scripts,
# not the multi-hundred-MB CSVs (those are for training, not serving)
COPY app.py .
COPY churniq_model.joblib .
COPY churniq_model_metadata.json .

EXPOSE 8000

# Basic container-level healthcheck hitting the API's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
