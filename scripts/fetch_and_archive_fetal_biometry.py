#!/usr/bin/env python3
"""
Automated Roboflow Data Ingestion & MinIO Archival Script
=========================================================
Project: Fetal Biometry Detection (Abdomen, Brain, Femur)
Author: Jacob Itopa (MLOps Platform)

Fetches the annotated fetal biometry ultrasound dataset from Roboflow in YOLOv11 format,
validates split integrity across classes, packages timestamped archives, and persists to MinIO S3.
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
logger = logging.getLogger("FetalBiometryDataIngest")

# ==============================================================================
# 1. Configuration & Environment Variables
# ==============================================================================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "276adphcEvBdxpqp05OI")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "liqourvolume")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "brain_femur_abdomen-iujfu")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "2"))
DATASET_FORMAT = os.getenv("DATASET_FORMAT", "yolov11")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio-ml-workshop:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "fetal-biometry-data")

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/tmp/fetal_data")


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


def ensure_bucket_exists(s3_client, bucket_name):
    """Checks if the destination MinIO bucket exists, creating it if necessary."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except Exception:
        logger.info(f"Bucket '{bucket_name}' not found. Creating bucket...")
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' successfully created.")


def find_dataset_root(search_dir):
    """Recursively locates the directory containing data.yaml."""
    for root, dirs, files in os.walk(search_dir):
        if "data.yaml" in files:
            return root
    return None


def download_from_roboflow(download_dir):
    """Downloads the dataset using the Roboflow Python SDK."""
    logger.info("Initializing Roboflow client...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    logger.info(f"Accessing project: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}...")
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    
    logger.info(f"Selecting version {ROBOFLOW_VERSION} (format: {DATASET_FORMAT})...")
    version = project.version(ROBOFLOW_VERSION)
    
    logger.info(f"Downloading dataset into: {download_dir}...")
    prev_cwd = os.getcwd()
    os.makedirs(download_dir, exist_ok=True)
    os.chdir(download_dir)
    try:
        dataset = version.download(DATASET_FORMAT)
        logger.info(f"Download complete! Dataset location reported: {dataset.location}")
    finally:
        os.chdir(prev_cwd)

    actual_root = find_dataset_root(download_dir)
    if not actual_root:
        raise FileNotFoundError(f"Could not locate data.yaml anywhere under {download_dir}")
    
    logger.info(f"Located verified dataset root: {actual_root}")
    return actual_root


def validate_dataset(dataset_dir):
    """Validates data.yaml and counts images in train, valid, and test splits."""
    logger.info(f"Validating dataset structure at: {dataset_dir}...")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml missing from downloaded dataset: {data_yaml_path}")
    
    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg.get("names", [])
    logger.info(f"Detected class labels: {classes} (count: {len(classes)})")
    
    stats = {
        "classes": classes,
        "splits": {},
        "total_images": 0
    }
    
    for split in ["train", "valid", "test", "val"]:
        split_img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.exists(split_img_dir):
            images = glob.glob(os.path.join(split_img_dir, "*.*"))
            valid_images = [img for img in images if img.lower().endswith((".jpg", ".jpeg", ".png"))]
            normalized_split = "valid" if split == "val" else split
            stats["splits"][normalized_split] = len(valid_images)
            stats["total_images"] += len(valid_images)
            logger.info(f"  • Split '{normalized_split}': {len(valid_images)} images")
    
    if stats["total_images"] == 0:
        raise ValueError("Dataset validation failed: 0 images found across all splits!")
    
    logger.info(f"Total dataset images verified: {stats['total_images']}")
    return stats


def package_and_archive(dataset_dir, output_archive_path):
    """Compresses dataset directory into a zip archive."""
    logger.info(f"Packaging dataset into archive: {output_archive_path}...")
    os.makedirs(os.path.dirname(output_archive_path), exist_ok=True)
    with zipfile.ZipFile(output_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, archive_name)
    archive_size_mb = os.path.getsize(output_archive_path) / (1024 * 1024)
    logger.info(f"Archive successfully created ({archive_size_mb:.2f} MB)")
    return archive_size_mb


def upload_dataset_directory(s3_client, local_dir, bucket_name, s3_prefix):
    """Recursively uploads an extracted directory structure to MinIO."""
    logger.info(f"Mirroring extracted dataset to s3://{bucket_name}/{s3_prefix}...")
    uploaded_files = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
            s3_key = f"{s3_prefix}/{rel_path}"
            s3_client.upload_file(local_path, bucket_name, s3_key)
            uploaded_files += 1
            if uploaded_files % 200 == 0:
                logger.info(f"  • Uploaded {uploaded_files} files...")
    logger.info(f"Uploaded total {uploaded_files} files to s3://{bucket_name}/{s3_prefix}")


def main():
    logger.info("=" * 65)
    logger.info("  Starting Fetal Biometry Data Ingestion Pipeline")
    logger.info("=" * 65)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    download_dir = os.path.join(WORKSPACE_DIR, "dataset_download")
    archive_filename = f"fetal_biometry_v{ROBOFLOW_VERSION}_{timestamp}.zip"
    archive_local_path = os.path.join(WORKSPACE_DIR, "archives", archive_filename)
    
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    
    s3 = get_s3_client()
    ensure_bucket_exists(s3, MINIO_BUCKET)
    
    try:
        # 1. Download
        dataset_location = download_from_roboflow(download_dir)
        
        # 2. Validate
        stats = validate_dataset(dataset_location)
        
        # 3. Create zip archive
        archive_size_mb = package_and_archive(dataset_location, archive_local_path)
        
        # 4. Upload zip archive to MinIO
        s3_archive_key = f"archives/{archive_filename}"
        logger.info(f"Uploading archive to s3://{MINIO_BUCKET}/{s3_archive_key}...")
        s3.upload_file(archive_local_path, MINIO_BUCKET, s3_archive_key)
        
        # 5. Mirror uncompressed dataset to S3 for direct training consumption
        s3_dataset_prefix = f"datasets/v{ROBOFLOW_VERSION}"
        upload_dataset_directory(s3, dataset_location, MINIO_BUCKET, s3_dataset_prefix)
        
        # 6. Upload latest manifest metadata
        manifest = {
            "project_name": ROBOFLOW_PROJECT,
            "workspace": ROBOFLOW_WORKSPACE,
            "version": ROBOFLOW_VERSION,
            "format": DATASET_FORMAT,
            "timestamp": timestamp,
            "archive_key": s3_archive_key,
            "archive_size_mb": round(archive_size_mb, 2),
            "dataset_prefix": s3_dataset_prefix,
            "total_images": stats["total_images"],
            "classes": stats["classes"],
            "splits": stats["splits"]
        }
        
        manifest_key = "datasets/latest_manifest.json"
        logger.info(f"Publishing manifest to s3://{MINIO_BUCKET}/{manifest_key}...")
        s3.put_object(
            Bucket=MINIO_BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        
        # Also store versioned manifest
        s3.put_object(
            Bucket=MINIO_BUCKET,
            Key=f"manifests/manifest_v{ROBOFLOW_VERSION}_{timestamp}.json",
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        
        logger.info("=" * 65)
        logger.info("🎉 Ingestion Pipeline Completed Successfully!")
        logger.info(f"   Bucket:      s3://{MINIO_BUCKET}")
        logger.info(f"   Archive:     {s3_archive_key} ({archive_size_mb:.2f} MB)")
        logger.info(f"   Total Scans: {stats['total_images']} images")
        logger.info(f"   Classes:     {stats['classes']}")
        logger.info("=" * 65)
        
    finally:
        if os.path.exists(WORKSPACE_DIR):
            shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
            logger.info("Cleaned up local temporary directory.")


if __name__ == "__main__":
    main()

