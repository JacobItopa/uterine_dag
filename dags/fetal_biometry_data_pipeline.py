"""
Fetal Biometry Weekly Data Ingestion DAG
========================================
Airflow DAG that executes once every week on Mondays at 09:00 UTC (0 9 * * 1).
Downloads the annotated Fetal Biometry (Abdomen, Brain, Femur) dataset from Roboflow in YOLOv11 format,
validates split integrity, packages a timestamped archive, and persists to MinIO S3 (bucket: fetal-biometry-data).

Repository: https://github.com/JacobItopa/uterine_dag.git
"""

from datetime import datetime, timedelta
from airflow import DAG

try:
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
except ImportError:
    try:
        from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
    except ImportError:
        from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator


default_args = {
    'owner': 'JacobItopa',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    dag_id='fetal_biometry_weekly_data_pipeline',
    default_args=default_args,
    description='Automated weekly Roboflow fetal biometry ultrasound ingestion & MinIO archival (Runs Mondays)',
    schedule='0 9 * * 1',
    catchup=False,
    tags=['medical-cv', 'fetal-biometry', 'yolov11', 'roboflow', 'minio', 'mlops'],
) as dag:

    # --------------------------------------------------------------------------
    # Task 1: Check MinIO Storage Readiness
    # --------------------------------------------------------------------------
    check_storage = KubernetesPodOperator(
        namespace='ml-workshop',
        image='curlimages/curl:latest',
        service_account_name='airflow-worker',
        cmds=['/bin/sh', '-c'],
        arguments=[
            'echo "Checking MinIO cluster connectivity..." && '
            'curl -sSf http://minio-ml-workshop:9000/minio/health/live && '
            'echo "MinIO is healthy and ready!"'
        ],
        name='check-minio-health-fetal',
        task_id='check_minio_storage_health',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # --------------------------------------------------------------------------
    # Task 2: Fetch Roboflow Dataset & Archive to MinIO (Isolated Container)
    # --------------------------------------------------------------------------
    fetch_and_archive = KubernetesPodOperator(
        namespace='ml-workshop',
        image='python:3.11-slim',
        service_account_name='airflow-worker',
        cmds=['/bin/bash', '-c'],
        arguments=[
            'set -e && '
            'echo "=== Step 1: Installing Dependencies ===" && '
            'apt-get update -qq && apt-get install -y -qq git curl > /dev/null 2>&1 && '
            'pip install --no-cache-dir -q roboflow boto3 pyyaml && '
            'echo "=== Step 2: Cloning Pipeline Repository ===" && '
            'rm -rf /tmp/uterine_dag && '
            'git clone --depth 1 https://github.com/JacobItopa/uterine_dag.git /tmp/uterine_dag && '
            'cd /tmp/uterine_dag && '
            'echo "=== Step 3: Executing Ingestion Script for Fetal Biometry ===" && '
            'python scripts/fetch_and_archive_fetal_biometry.py && '
            'echo "=== Fetal Biometry Ingestion & Archival Succeeded! ==="'
        ],
        env_vars={
            'ROBOFLOW_API_KEY': '276adphcEvBdxpqp05OI',
            'ROBOFLOW_WORKSPACE': 'liqourvolume',
            'ROBOFLOW_PROJECT': 'brain_femur_abdomen-iujfu',
            'ROBOFLOW_VERSION': '2',
            'DATASET_FORMAT': 'yolov11',
            'MINIO_ENDPOINT': 'http://minio-ml-workshop:9000',
            'MINIO_ACCESS_KEY': 'minio',
            'MINIO_SECRET_KEY': 'minio123',
            'MINIO_BUCKET': 'fetal-biometry-data',
        },
        security_context={'run_as_user': 0},
        labels={'app': 'fetal-biometry-ingest'},
        name='fetal-data-fetch-task',
        task_id='fetch_and_archive_roboflow_dataset',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # --------------------------------------------------------------------------
    # Task 3: Ingestion Audit & Manifest Verification
    # --------------------------------------------------------------------------
    audit_manifest = KubernetesPodOperator(
        namespace='ml-workshop',
        image='python:3.11-slim',
        service_account_name='airflow-worker',
        cmds=['/bin/bash', '-c'],
        arguments=[
            'pip install --no-cache-dir -q boto3 && '
            'python -c "'
            'import boto3, json\n'
            's3 = boto3.client(\'s3\', endpoint_url=\'http://minio-ml-workshop:9000\', aws_access_key_id=\'minio\', aws_secret_access_key=\'minio123\')\n'
            'obj = s3.get_object(Bucket=\'fetal-biometry-data\', Key=\'datasets/latest_manifest.json\')\n'
            'manifest = json.loads(obj[\'Body\'].read().decode(\'utf-8\'))\n'
            'print(\'=== FETAL BIOMETRY INGESTION AUDIT REPORT ===\')\n'
            'print(json.dumps(manifest, indent=2))\n'
            'print(\'Total Scans Verified:\', manifest.get(\'total_images\'))\n'
            'print(\'Classes Detected:\', manifest.get(\'classes\'))\n'
            'print(\'Archive Location:\', manifest.get(\'archive_key\'))\n'
            '"'
        ],
        name='audit-fetal-manifest',
        task_id='audit_latest_manifest',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Task Pipeline Dependencies
    check_storage >> fetch_and_archive >> audit_manifest
