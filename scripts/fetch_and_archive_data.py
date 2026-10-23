#!/usr/bin/env python3
"""
Automated Roboflow Data Ingestion & MinIO Archival Script
=========================================================
Project: Uterine Fibroid Detection (Medical Computer Vision)
Author: Jacob Itopa (MLOps Platform)

Fetches the latest annotated ultrasound dataset from Roboflow in YOLOv11 format,
validates split integrity, packages timestamped archives, and persists to MinIO S3.
"""

import os
import sys
import json
import glob
import shutil
import zipfile
import logging
from datetime import datetime, timezone
import yaml
import boto3
from botocore.client import Config
from roboflow import Roboflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("UterineDataIngest")

# ==============================================================================
# 1. Configuration & Environment Variables
# ==============================================================================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "276adphcEvBdxpqp05OI")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "liqourvolume")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "uterine-fibroid-oswew-wphbc")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
DATASET_FORMAT = os.getenv("DATASET_FORMAT", "yolov11")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio-ml-workshop:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "fibroid-data")

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/uterine_data")


def get_s3_client():
    """Initializes and returns a Boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )


def ensure_minio_bucket(s3_client, bucket_name):
    """Ensures that the target MinIO bucket exists."""
    logger.info(f"Checking MinIO bucket: '{bucket_name}' at {MINIO_ENDPOINT}...")
    try:
        response = s3_client.list_buckets()
        buckets = [b["Name"] for b in response.get("Buckets", [])]
        if bucket_name not in buckets:
            logger.info(f"Bucket '{bucket_name}' not found. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to verify/create MinIO bucket '{bucket_name}': {e}")
        raise


def find_dataset_root(location):
    """Finds the directory containing data.yaml."""
    if os.path.exists(os.path.join(location, "data.yaml")):
        return location
    for root, dirs, files in os.walk(location):
        if "data.yaml" in files:
            logger.info(f"Resolved dataset root to: {root}")
            return root
    return location


def fetch_roboflow_dataset(download_dir):
    """Downloads the dataset from Roboflow in YOLOv11 format."""
    logger.info("Initializing Roboflow client...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    logger.info(f"Accessing project: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}...")
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    
    logger.info(f"Selecting version {ROBOFLOW_VERSION} (format: {DATASET_FORMAT})...")
    version = project.version(ROBOFLOW_VERSION)
    
    os.makedirs(download_dir, exist_ok=True)
    orig_cwd = os.getcwd()
    try:
        os.chdir(download_dir)
        logger.info(f"Downloading dataset in: {download_dir}...")
        dataset = version.download(DATASET_FORMAT)
        resolved_location = find_dataset_root(dataset.location)
    finally:
        os.chdir(orig_cwd)
    
    logger.info(f"Download complete! Dataset location: {resolved_location}")
    return resolved_location


def validate_dataset(dataset_dir):
    """Validates the downloaded dataset structure and counts images/classes."""
    dataset_dir = find_dataset_root(dataset_dir)
    logger.info(f"Validating dataset structure at: {dataset_dir}...")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml missing from downloaded dataset: {data_yaml_path}")

    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    classes = config.get("names", [])
    num_classes = config.get("nc", len(classes))
    logger.info(f"Configured Classes ({num_classes}): {classes}")

    train_imgs = glob.glob(os.path.join(dataset_dir, "train", "images", "*"))
    val_imgs = glob.glob(os.path.join(dataset_dir, "valid", "images", "*"))
    test_imgs = glob.glob(os.path.join(dataset_dir, "test", "images", "*"))

    total_images = len(train_imgs) + len(val_imgs) + len(test_imgs)
    stats = {
        "train_count": len(train_imgs),
        "valid_count": len(val_imgs),
        "test_count": len(test_imgs),
        "total_images": total_images,
        "classes": classes,
        "nc": num_classes
    }

    logger.info(f"Dataset Counts -> Train: {len(train_imgs)}, Valid: {len(val_imgs)}, Test: {len(test_imgs)}, Total: {total_images}")
    if total_images == 0:
        raise ValueError("Downloaded dataset contains 0 images across all splits!")

    return stats


def package_and_archive(dataset_dir, output_zip_path):
    """Creates a clean compressed zip archive of the dataset."""
    logger.info(f"Archiving dataset to {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, dataset_dir)
                zipf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    logger.info(f"Archive created successfully: {output_zip_path} ({zip_size_mb:.2f} MB)")
    return zip_size_mb


def upload_to_minio(s3_client, bucket, zip_path, dataset_dir, stats):
    """Uploads timestamped archive, latest pointer, and metadata manifest to MinIO."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_tag = f"v{ROBOFLOW_VERSION}"

    # 1. Timestamped Archive
    raw_key = f"raw/uterine_fibroid_{version_tag}_{timestamp}.zip"
    logger.info(f"Uploading timestamped archive to s3://{bucket}/{raw_key}...")
    s3_client.upload_file(zip_path, bucket, raw_key)

    # 2. Latest Pointer Archive
    latest_key = "dataset/latest/uterine_fibroid_latest.zip"
    logger.info(f"Uploading latest pointer to s3://{bucket}/{latest_key}...")
    s3_client.upload_file(zip_path, bucket, latest_key)

    # 3. YAML Configuration
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_key = "dataset/latest/data.yaml"
    logger.info(f"Uploading data.yaml to s3://{bucket}/{yaml_key}...")
    s3_client.upload_file(data_yaml_path, bucket, yaml_key)

    # 4. Ingestion Manifest Metadata
    manifest = {
        "pipeline": "uterine_fibroid_weekly_data_ingest",
        "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "roboflow_source": {
            "workspace": ROBOFLOW_WORKSPACE,
            "project": ROBOFLOW_PROJECT,
            "version": ROBOFLOW_VERSION,
            "format": DATASET_FORMAT
        },
        "statistics": stats,
        "storage": {
            "bucket": bucket,
            "raw_archive_key": raw_key,
            "latest_archive_key": latest_key,
            "data_yaml_key": yaml_key
        }
    }
    manifest_path = "/tmp/ingestion_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest_key = f"metadata/ingestion_manifest_{timestamp}.json"
    latest_manifest_key = "metadata/ingestion_manifest_latest.json"
    logger.info(f"Uploading manifest to s3://{bucket}/{manifest_key}...")
    s3_client.upload_file(manifest_path, bucket, manifest_key)
    s3_client.upload_file(manifest_path, bucket, latest_manifest_key)

    logger.info("All files uploaded to MinIO successfully!")
    return manifest


def main():
    logger.info("=================================================================")
    logger.info("  🚀 Starting Uterine Fibroid Weekly Data Ingestion Pipeline")
    logger.info("=================================================================")

    s3_client = get_s3_client()
    ensure_minio_bucket(s3_client, MINIO_BUCKET)

    download_dir = os.path.join(WORKSPACE_DIR, "dataset_download")
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    try:
        # Download
        dataset_location = fetch_roboflow_dataset(download_dir)

        # Validate
        stats = validate_dataset(dataset_location)

        # Archive
        zip_path = os.path.join(WORKSPACE_DIR, f"uterine_fibroid_v{ROBOFLOW_VERSION}.zip")
        package_and_archive(dataset_location, zip_path)

        # Upload
        manifest = upload_to_minio(s3_client, MINIO_BUCKET, zip_path, dataset_location, stats)

        logger.info("=================================================================")
        logger.info("  🎉 INGESTION PIPELINE FINISHED SUCCESSFULLY")
        logger.info(f"  • Total Ultrasound Images: {stats['total_images']}")
        logger.info(f"  • Classes: {stats['classes']}")
        logger.info(f"  • Raw S3 Key: {manifest['storage']['raw_archive_key']}")
        logger.info(f"  • Latest S3 Key: {manifest['storage']['latest_archive_key']}")
        logger.info("=================================================================")
    finally:
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
            logger.info("Cleaned up local temporary directory.")


if __name__ == "__main__":
    main()
