#!/usr/bin/env python3
"""
Automated Model Serving Container Builder (Kaniko & MLflow)
===========================================================
Downloads the trained YOLOv11 model weights from MinIO / MLflow,
sets up Docker authentication, and builds & pushes the Seldon inference
container to Docker Hub using Google Kaniko inside Kubernetes.
"""

import os
import sys
import json
import base64
import subprocess
import boto3
from botocore.client import Config

# Configuration from Environment Variables
CONTAINER_REGISTRY_USER = os.environ.get("CONTAINER_REGISTRY_USER", "y2jay1")
CONTAINER_REGISTRY_PASSWORD = os.environ.get("CONTAINER_REGISTRY_PASSWORD", "#Jacob123")
CONTAINER_DETAILS = os.environ.get("CONTAINER_DETAILS", "y2jay1/uterine-fibroid-detector:v2")

MINIO_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio-ml-workshop:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

MODEL_NAME = os.environ.get("MODEL_NAME", "Uterine-Fibroid-YOLOv11")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2")


def setup_docker_credentials():
    """Configures Kaniko's /kaniko/.docker/config.json with Docker Hub auth."""
    print("=== Step 1: Configuring Docker Hub Credentials for Kaniko ===")
    auth_str = f"{CONTAINER_REGISTRY_USER}:{CONTAINER_REGISTRY_PASSWORD}"
    encoded_creds = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")

    config_data = {
        "auths": {
            "https://index.docker.io/v1/": {"auth": encoded_creds},
            "index.docker.io": {"auth": encoded_creds},
            "docker.io": {"auth": encoded_creds}
        }
    }

    os.makedirs("/kaniko/.docker", exist_ok=True)
    with open("/kaniko/.docker/config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print("Docker config successfully written to /kaniko/.docker/config.json")


def download_model_weights(destination_path):
    """Downloads model weights from MinIO S3 storage."""
    print("=== Step 2: Downloading Model Weights from MinIO ===")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

    candidate_keys = []

    # 1. Try querying MLflow tracking server for exact model source URI
    mlflow_host = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5500")
    try:
        import urllib.request
        url = f"{mlflow_host}/api/2.0/mlflow/model-versions/get?name={MODEL_NAME}&version={MODEL_VERSION}"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "model_version" in data:
                src = data["model_version"].get("source", "")
                print(f"MLflow model version {MODEL_VERSION} source: {src}")
                if src.startswith("s3://"):
                    parts = src[5:].split("/", 1)
                    if len(parts) == 2:
                        candidate_keys.append((parts[0], parts[1]))
    except Exception as e:
        print(f"MLflow API lookup skipped: {e}")

    # 2. Known default candidates
    candidate_keys.extend([
        ("mlflow", "1/ec4df09f09a240239ecae365641a7090/artifacts/weights/best.pt"),
        ("fibroid-data", "models/yolov11n_best.pt"),
        ("fibroid-data", "models/yolov11n_test_run.pt"),
    ])

    downloaded = False
    for bucket, key in candidate_keys:
        try:
            print(f"Checking s3://{bucket}/{key}...")
            s3.head_object(Bucket=bucket, Key=key)
            print(f"Found weights at s3://{bucket}/{key}! Downloading to {destination_path}...")
            s3.download_file(bucket, key, destination_path)
            file_size_mb = os.path.getsize(destination_path) / (1024 * 1024)
            print(f"Successfully downloaded weights: {file_size_mb:.2f} MB")
            downloaded = True
            break
        except Exception as e:
            print(f"Not found in s3://{bucket}/{key}: {e}")

    if not downloaded:
        print("Searching for any .pt weights in fibroid-data bucket...")
        resp = s3.list_objects_v2(Bucket="fibroid-data", Prefix="models/")
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".pt"):
                print(f"Found alternative: s3://fibroid-data/{obj['Key']}. Downloading...")
                s3.download_file("fibroid-data", obj["Key"], destination_path)
                downloaded = True
                break

    if not downloaded or not os.path.exists(destination_path):
        raise FileNotFoundError("Could not find or download model weights from MinIO!")


def run_kaniko_build():
    """Executes the Kaniko executor to build and push the container."""
    print("=== Step 3: Executing Kaniko Container Build & Push ===")
    kaniko_cmd = [
        "/kaniko/executor",
        "--context=/microservice",
        "--dockerfile=/microservice/Dockerfile",
        f"--destination={CONTAINER_DETAILS}",
        "--cache=true",
        "--compressed-caching=false"
    ]

    print("Running command:", " ".join(kaniko_cmd))
    proc = subprocess.run(kaniko_cmd, check=True)
    print("=" * 60)
    print(f"🎉 Successfully built and pushed: {CONTAINER_DETAILS}")
    print("=" * 60)


def main():
    setup_docker_credentials()

    dest_weights = "/microservice/best.pt"
    os.makedirs("/microservice", exist_ok=True)
    import shutil
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(current_dir):
        src = os.path.join(current_dir, f)
        if os.path.isfile(src) and not f.endswith(".pt"):
            print(f"Copying {f} to /microservice/...")
            shutil.copy(src, os.path.join("/microservice", f))

    download_model_weights(dest_weights)
    run_kaniko_build()


if __name__ == "__main__":
    main()
