#!/usr/bin/env python3
"""
Uterine Fibroid Detection: Live Inference Client & Benchmark Probe
==================================================================
Sends ultrasound scan frames to the deployed Seldon Core inference service,
benchmarks latency, and parses detected fibroid bounding boxes.

Usage:
    python test_inference.py --image path/to/ultrasound.jpg
    python test_inference.py --url http://fibroid-model.192.168.49.2.nip.io/api/v1.0/predictions
"""

import os
import sys
import json
import time
import base64
import argparse
import urllib.request
import urllib.error

DEFAULT_URL = "http://fibroid-model.192.168.49.2.nip.io/api/v1.0/predictions"


def encode_image(image_path):
    """Encodes an image file into base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def send_prediction(url, image_b64, timeout=10.0):
    """Sends inference request to Seldon endpoint."""
    payload = {
        "data": {
            "ndarray": [image_b64]
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.time() - start_time) * 1000.0
            body = json.loads(resp.read().decode("utf-8"))
            return {
                "success": True,
                "status_code": resp.getcode(),
                "latency_ms": latency_ms,
                "response": body
            }
    except urllib.error.HTTPError as e:
        latency_ms = (time.time() - start_time) * 1000.0
        err_body = e.read().decode("utf-8") if e.fp else str(e)
        return {
            "success": False,
            "status_code": e.code,
            "latency_ms": latency_ms,
            "error": err_body
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "success": False,
            "status_code": None,
            "latency_ms": latency_ms,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Test Uterine Fibroid Detection Endpoint")
    parser.add_argument("--url", default=DEFAULT_URL, help="Inference URL endpoint")
    parser.add_argument("--image", default=None, help="Path to ultrasound image file (JPEG/PNG)")
    parser.add_argument("--count", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        print(f"Loading image from: {args.image}")
        b64_data = encode_image(args.image)
    else:
        # Fallback 1x1 dummy jpeg frame
        print("Using synthetic ultrasound probe frame...")
        b64_data = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="

    print("=" * 70)
    print(f"Target Service: {args.url}")
    print(f"Running {args.count} inference benchmark iterations...")
    print("=" * 70)

    latencies = []
    successes = 0

    for i in range(args.count):
        res = send_prediction(args.url, b64_data, timeout=args.timeout)
        if res["success"]:
            successes += 1
            latencies.append(res["latency_ms"])
            data_body = res["response"].get("data", {})
            detections = data_body.get("ndarray", [])
            names = data_body.get("names", ["cls_id", "label", "conf", "x1", "y1", "x2", "y2"])

            print(f"[{i+1}/{args.count}] HTTP {res['status_code']} | Latency: {res['latency_ms']:.1f}ms")
            print(f"    Detection Columns: {names}")
            for d in detections:
                print(f"    -> {d}")
        else:
            print(f"[{i+1}/{args.count}] FAILED | HTTP {res['status_code']} | Error: {res.get('error')}")

        time.sleep(0.3)

    print("=" * 70)
    print("Benchmark Summary:")
    print(f"Total Probes:   {args.count}")
    print(f"Success Rate:   {successes}/{args.count} ({successes/args.count*100:.1f}%)")
    if latencies:
        print(f"Latency Min:    {min(latencies):.1f} ms")
        print(f"Latency Avg:    {sum(latencies)/len(latencies):.1f} ms")
        print(f"Latency Max:    {max(latencies):.1f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
