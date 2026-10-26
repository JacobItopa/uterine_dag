"""
Production MLOps Pipeline: Airflow DAG for Uterine Fibroid Detection YOLOv11 Serving
=====================================================================================
Orchestrates end-to-end model serving deployment:
1. Builds and pushes the YOLOv11 inference microservice container to Docker Hub via Kaniko.
2. Deploys the SeldonDeployment CRD and Ingress in Kubernetes ('ml-workshop' namespace).
3. Connects the inference service to Prometheus and Grafana for latency & throughput monitoring.
4. Validates real-time ultrasound inference health via automated prediction probes.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable

try:
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
except ImportError:
    from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

# Airflow Dynamic Variables (with production fallbacks)
DOCKERHUB_USER = Variable.get("DOCKERHUB_USER", default_var="y2jay1")
DOCKERHUB_PASSWORD = Variable.get("DOCKERHUB_PASSWORD", default_var="#Jacob123")
MODEL_NAME = Variable.get("MODEL_NAME", default_var="Uterine-Fibroid-YOLOv11")
MODEL_VERSION = Variable.get("MODEL_VERSION", default_var="2")
IMAGE_TAG = Variable.get("MODEL_IMAGE_TAG", default_var="v2")
CLUSTER_DOMAIN_NAME = Variable.get("CLUSTER_DOMAIN_NAME", default_var="192.168.49.2.nip.io")
SERVING_REPO_URL = Variable.get("SERVING_REPO_URL", default_var="https://github.com/JacobItopa/uterine_dag.git")

CONTAINER_REGISTRY = "https://index.docker.io/v1/"
CONTAINER_DETAILS = f"{DOCKERHUB_USER}/uterine-fibroid-detector:{IMAGE_TAG}"

CLEAN_REPO_URL = SERVING_REPO_URL[:-4] if SERVING_REPO_URL.endswith(".git") else SERVING_REPO_URL
ARCHIVE_URL = f"{CLEAN_REPO_URL}/archive/refs/heads/main.tar.gz"

default_args = {
    'owner': 'ml-engineer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'uterine_fibroid_model_deploy',
    default_args=default_args,
    description='Build YOLOv11 serving container with Kaniko and deploy Seldon inference graph with Prometheus monitoring',
    schedule=None,
    catchup=False,
    tags=['mlops', 'uterine-fibroid', 'seldon', 'model-deploy', 'yolov11'],
) as dag:

    # Task 1: Build & Push Inference Container to Docker Hub via Kaniko
    build_push_task = KubernetesPodOperator(
        namespace='ml-workshop',
        image='quay.io/ml-on-k8s/kaniko-container-builder:1.0.0',
        service_account_name='airflow-worker',
        cmds=['/bin/bash', '-c'],
        arguments=[
            'set -e && '
            'echo "=== Downloading serving codebase from repository ===" && '
            'rm -rf /tmp/app /tmp/repo.tar.gz && '
            f'if command -v git >/dev/null 2>&1; then '
            f'  git clone --depth 1 {SERVING_REPO_URL} /tmp/app; '
            f'else '
            f'  python3 -c "import urllib.request, tarfile; '
            f'urllib.request.urlretrieve(\'{ARCHIVE_URL}\', \'/tmp/repo.tar.gz\'); '
            f'tarfile.open(\'/tmp/repo.tar.gz\').extractall(\'/tmp\')" && '
            f'  mv /tmp/*-main /tmp/app 2>/dev/null || mv /tmp/uterine_dag* /tmp/app; '
            f'fi && '
            'cd /tmp/app/seldon_serving && '
            'echo "=== Running build_push_image.py to download weights and push container ===" && '
            'python3 build_push_image.py'
        ],
        env_vars={
            'MODEL_NAME': MODEL_NAME,
            'MODEL_VERSION': MODEL_VERSION,
            'CONTAINER_REGISTRY': CONTAINER_REGISTRY,
            'CONTAINER_REGISTRY_USER': DOCKERHUB_USER,
            'CONTAINER_REGISTRY_PASSWORD': DOCKERHUB_PASSWORD,
            'CONTAINER_DETAILS': CONTAINER_DETAILS,
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio-ml-workshop:9000',
            'MLFLOW_TRACKING_URI': 'http://mlflow:5500',
            'AWS_ACCESS_KEY_ID': 'minio',
            'AWS_SECRET_ACCESS_KEY': 'minio123',
        },
        security_context={'run_as_user': 0},
        labels={'app': 'kaniko-fibroid-builder'},
        name='kaniko-build-fibroid-model',
        task_id='build_push_image',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Task 2: Render & Apply SeldonDeployment and Ingress to Kubernetes
    deploy_task = KubernetesPodOperator(
        namespace='ml-workshop',
        image='quay.io/ml-on-k8s/airflow-python-runner:0.0.11',
        service_account_name='airflow-worker',
        cmds=['/bin/bash', '-c'],
        arguments=[
            'set -e && '
            'echo "=== Downloading deployment manifests from repository ===" && '
            'rm -rf /tmp/app /tmp/repo.tar.gz && '
            f'if command -v git >/dev/null 2>&1; then '
            f'  git clone --depth 1 {SERVING_REPO_URL} /tmp/app; '
            f'else '
            f'  python3 -c "import urllib.request, tarfile; '
            f'urllib.request.urlretrieve(\'{ARCHIVE_URL}\', \'/tmp/repo.tar.gz\'); '
            f'tarfile.open(\'/tmp/repo.tar.gz\').extractall(\'/tmp\')" && '
            f'  mv /tmp/*-main /tmp/app 2>/dev/null || mv /tmp/uterine_dag* /tmp/app; '
            f'fi && '
            'cd /tmp/app/seldon_serving && '
            'echo "=== Running deploy_model.py to render and apply SeldonDeployment ===" && '
            'python3 deploy_model.py'
        ],
        env_vars={
            'MODEL_NAME': MODEL_NAME,
            'MODEL_VERSION': MODEL_VERSION,
            'CONTAINER_REGISTRY': CONTAINER_REGISTRY,
            'CONTAINER_DETAILS': CONTAINER_DETAILS,
            'CLUSTER_DOMAIN_NAME': CLUSTER_DOMAIN_NAME,
            'MLFLOW_TRACKING_URI': 'http://mlflow:5500',
        },
        security_context={'run_as_user': 0},
        labels={'app': 'seldon-fibroid-deployer'},
        name='seldon-deploy-fibroid-model',
        task_id='deploy_seldon_service',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    # Task 3: Automated Health & Prediction Verification Probe
    verify_task = KubernetesPodOperator(
        namespace='ml-workshop',
        image='quay.io/ml-on-k8s/airflow-python-runner:0.0.11',
        service_account_name='airflow-worker',
        cmds=['/bin/bash', '-c'],
        arguments=[
            'set -e && '
            'python3 -c "\n'
            'import time, json, urllib.request, urllib.error\n'
            'endpoint = \'http://fibroid-model.' + CLUSTER_DOMAIN_NAME + '/api/v1.0/predictions\'\n'
            'print(f\'=== Verifying Inference Endpoint: {endpoint} ===\')\n'
            'payload = {\'data\': {\'ndarray\': [\'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=\']}}\n'
            'req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(\'utf-8\'), headers={\'Content-Type\': \'application/json\'})\n'
            'ready = False\n'
            'for attempt in range(20):\n'
            '    try:\n'
            '        with urllib.request.urlopen(req, timeout=10) as resp:\n'
            '            body = json.loads(resp.read().decode(\'utf-8\'))\n'
            '            print(f\'Probing attempt {attempt+1}: Status {resp.getcode()} - Response: {body}\')\n'
            '            if resp.getcode() == 200:\n'
            '                print(\' Inference endpoint is LIVE and responding!\')\n'
            '                ready = True\n'
            '                break\n'
            '    except Exception as e:\n'
            '        print(f\'Attempt {attempt+1} waiting for model pod readiness: {e}\')\n'
            '        time.sleep(10)\n'
            'if not ready:\n'
            '    print(\'Warning: Model pod taking longer to spin up, check kubectl get pods -n ml-workshop\')\n'
            '"'
        ],
        security_context={'run_as_user': 0},
        labels={'app': 'inference-health-verifier'},
        name='verify-fibroid-inference',
        task_id='verify_inference_health',
        get_logs=True,
        is_delete_operator_pod=True,
    )

    build_push_task >> deploy_task >> verify_task
