# Containerized inference API for the DR screening pipeline.
# Build:  docker build -t dr-screening-api .
# Run:    docker run -p 8000:8000 dr-screening-api
# Docs:   http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# System deps needed by opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stage1_quality_assessment.py stage5_segmentation_train.py \
     stage6_classification_train.py stage7_pipeline.py api.py ./
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
