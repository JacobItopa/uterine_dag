#!/usr/bin/env python3
"""
Automated Seldon Deployment Script for Fetal Biometry Detection
==============================================================
Renders SeldonDeploy.yaml and Ingress.yaml using Jinja2 templates,
applies them to the Kubernetes cluster in 'ml-workshop' namespace,
and verifies service availability.
"""

import os
import sys
import time
import json
import urllib.request
from jinja2 import Template

try:
    import openshift as oc
except ImportError:
    oc = None

HOST = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5500")
MODEL_NAME = os.environ.get("MODEL_NAME", "Fetal-Biometry-YOLOv11")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
CLUSTER_DNS = os.environ.get("CLUSTER_DOMAIN_NAME", "192.168.49.2.nip.io")
CONTAINER_DETAILS = os.environ.get("CONTAINER_DETAILS", "y2jay1/fetal-biometry-detector:v1")


def get_model_run_info(host, model_name, model_version):
    """Retrieves run_id and experiment_id from MLflow tracking server."""
    if os.environ.get("RUN_ID") and os.environ.get("EXPERIMENT_ID"):
        return os.environ["RUN_ID"], os.environ["EXPERIMENT_ID"]

    def _fetch_json(url, post_data=None):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            body = json.dumps(post_data).encode("utf-8") if post_data is not None else None
            with urllib.request.urlopen(req, data=body, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Fetch failed for {url}: {e}")
            return None

    # 1. Try direct REST endpoint for model version
    url = f"{host}/api/2.0/mlflow/model-versions/get?name={model_name}&version={model_version}"
    data = _fetch_json(url)
    if data and "model_version" in data:
        mv = data["model_version"]
        run_id = mv.get("run_id")
        source = mv.get("source", "")
        exp_id = source.split("/")[3] if len(source.split("/")) > 3 else "2"
        print(f"Direct match found: run_id={run_id}, experiment_id={exp_id}")
        return run_id, exp_id

    # 2. Search all registered models
    print("Searching all registered models in MLflow...")
    search_data = _fetch_json(f"{host}/api/2.0/mlflow/registered-models/search")
    if search_data and "registered_models" in search_data:
        for m in search_data["registered_models"]:
            if m.get("name", "").lower() == model_name.lower():
                for lv in m.get("latest_versions", []):
                    if str(lv.get("version")) == str(model_version):
                        run_id = lv.get("run_id")
                        source = lv.get("source", "")
                        exp_id = source.split("/")[3] if len(source.split("/")) > 3 else "2"
                        print(f"Found model version {model_version}: run_id={run_id}, exp_id={exp_id}")
                        return run_id, exp_id

    # 3. Fallback: Search latest active run in experiment
    print("Fallback to latest run in MLflow...")
    runs_data = _fetch_json(f"{host}/api/2.0/mlflow/runs/search", post_data={"experiment_ids": ["2", "1", "0"], "max_results": 5})
    if runs_data and "runs" in runs_data and len(runs_data["runs"]) > 0:
        latest = runs_data["runs"][0]
        run_id = latest["info"]["run_id"]
        exp_id = latest["info"].get("experiment_id", "2")
        return run_id, exp_id

    # Default fallback
    print("Using default fallback run ID...")
    return "fetal-biometry-v1", "2"


def main():
    print("=" * 60)
    print("Deploying Seldon Core Inference Service: Fetal Biometry")
    print("=" * 60)

    run_id, experiment_id = get_model_run_info(HOST, MODEL_NAME, MODEL_VERSION)
    print(f"Model: {MODEL_NAME} (v{MODEL_VERSION})")
    print(f"Run ID: {run_id}")
    print(f"Experiment ID: {experiment_id}")

    clean_model_name = "fetal-biometry-detector"
    svc_name = f"model-{run_id[:8]}-fetal"
    ingress_host = f"{clean_model_name}.{CLUSTER_DNS}"

    print(f"Target Service Name: {svc_name}")
    print(f"Primary Ingress Host: {ingress_host}")
    print(f"Alias Ingress Host:   fetal-model.{CLUSTER_DNS}")
    print(f"Container Image:      {CONTAINER_DETAILS}")

    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

    token = None
    namespace = "ml-workshop"

    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    if os.path.exists(ns_path):
        with open(ns_path, "r") as f:
            namespace = f.read().strip()

    k8s_host = os.environ.get("KUBERNETES_SERVICE_HOST")
    k8s_port = os.environ.get("KUBERNETES_SERVICE_PORT")
    server = f"https://{k8s_host}:{k8s_port}" if k8s_host and k8s_port else "https://kubernetes.default.svc"

    template_data = {
        "experiment_id": run_id,
        "model_name": MODEL_NAME,
        "clean_model_name": clean_model_name,
        "svc_name": svc_name,
        "model_coordinates": CONTAINER_DETAILS,
        "ingress_host": ingress_host,
        "cluster_dns": CLUSTER_DNS,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    seldon_template = os.path.join(current_dir, "SeldonDeploy.yaml")
    ingress_template = os.path.join(current_dir, "Ingress.yaml")

    with open(seldon_template, "r") as f:
        rendered_seldon = Template(f.read()).render(template_data)

    with open(ingress_template, "r") as f:
        rendered_ingress = Template(f.read()).render(template_data)

    print("\n=== Rendered SeldonDeploy.yaml ===")
    print(rendered_seldon)

    print("\n=== Rendered Ingress.yaml ===")
    print(rendered_ingress)

    if oc is not None and token:
        print(f"\nApplying manifests to namespace '{namespace}' via OpenShift/Kubernetes client...")
        with oc.api_server(server):
            with oc.token(token):
                with oc.project(namespace), oc.timeout(10 * 60):
                    print("Applying SeldonDeployment...")
                    oc.apply(rendered_seldon)
                    print(f"SeldonDeployment '{svc_name}' applied successfully.")

                    print(f"Waiting for backing service '{svc_name}' to become available...")
                    max_attempts = 30
                    for attempt in range(max_attempts):
                        count = oc.selector(f"service/{svc_name}").count_existing()
                        if count > 0:
                            print(f"Service '{svc_name}' is ready!")
                            break
                        print(f"Service '{svc_name}' not ready yet (attempt {attempt + 1}/{max_attempts}), waiting 5s...")
                        time.sleep(5)

                    print("Applying Ingress...")
                    oc.apply(rendered_ingress)
                    print(f"Ingress for '{svc_name}' applied successfully.")
    else:
        print("\nApplying manifests via kubectl CLI...")
        import subprocess
        tmp_seldon = "/tmp/rendered_fetal_seldon.yaml"
        tmp_ingress = "/tmp/rendered_fetal_ingress.yaml"
        with open(tmp_seldon, "w") as f:
            f.write(rendered_seldon)
        with open(tmp_ingress, "w") as f:
            f.write(rendered_ingress)

        subprocess.run(["kubectl", "apply", "-f", tmp_seldon, "-n", namespace], check=True)
        print(f"Waiting for backing service '{svc_name}'...")
        for _ in range(30):
            res = subprocess.run(["kubectl", "get", "service", svc_name, "-n", namespace], capture_output=True)
            if res.returncode == 0:
                print(f"Service '{svc_name}' found!")
                break
            time.sleep(5)
        subprocess.run(["kubectl", "apply", "-f", tmp_ingress, "-n", namespace], check=True)
        print("Ingress applied successfully via kubectl.")

    print("\n" + "=" * 60)
    print(" Deployment Complete!")
    print(f" Prediction Endpoint: http://fetal-model.{CLUSTER_DNS}/api/v1.0/predictions")
    print(f" Prometheus Metrics:  http://fetal-model.{CLUSTER_DNS}/prometheus")
    print("=" * 60)


if __name__ == "__main__":
    main()

