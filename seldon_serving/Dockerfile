FROM python:3.11-slim-bookworm

WORKDIR /microservice

# Install system libraries for OpenCV and headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY base_requirements.txt /microservice/
RUN pip install --no-cache-dir -r base_requirements.txt

COPY *.py *.pt* /microservice/

ENV MODEL_NAME=Predictor
ENV SERVICE_TYPE=MODEL
ENV HTTP_PORT=9000
ENV METRICS_PORT=6000
ENV GRPC_PORT=5000

EXPOSE 9000 6000 5000

CMD seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --grpc-port ${GRPC_PORT} --metrics-port ${METRICS_PORT} --http-port ${HTTP_PORT} --log-level INFO
