# 🩺 Uterine Fibroid Weekly Data Ingestion Pipeline

An automated Kubernetes-native Apache Airflow pipeline for continuous ultrasound scan ingestion, validation, and object storage archival for the **Uterine Fibroid Detection** computer vision project.

---

## 🎯 Architecture & Data Flow

```
+---------------------------+
| Roboflow API              |
| (liqourvolume/            |
|  uterine-fibroid-oswew)   |
+-------------+-------------+
              |
              | (Every Monday at 09:00 UTC)
              v
+-------------+-------------+
| Airflow DAG               |
| uterine_fibroid_weekly_   |
| data_pipeline             |
+-------------+-------------+
              |
              | KubernetesPodOperator (python:3.11-slim)
              v
+-------------+-------------+
| Ingestion & Quality Engine|
| scripts/fetch_and_archive |
+-------------+-------------+
              |
              | Validates YOLOv11 & archives
              v
+---------------------------+
| MinIO S3 Object Storage   |
| Bucket: fibroid-data      |
|  • raw/v1_timestamp.zip   |
|  • dataset/latest/        |
|  • metadata/manifest.json |
+---------------------------+
```

---

## 🗓️ Schedule

- **Cron Expression:** `0 9 * * 1`
- **Schedule:** Once every week on **Mondays at 09:00 AM UTC**
- **Catchup:** `False`

---

## 📂 Repository Structure

```
uterine_dag/
├── .gitignore
├── README.md
├── requirements.txt
├── dags/
│   └── uterine_fibroid_data_pipeline.py   # Airflow DAG definition
└── scripts/
    └── fetch_and_archive_data.py          # Standalone ingestion & archival engine
```

---

## ⚙️ Environment Variables Configured

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ROBOFLOW_API_KEY` | `276adphcEvBdxpqp05OI` | Roboflow authentication token |
| `ROBOFLOW_WORKSPACE`| `liqourvolume` | Roboflow workspace slug |
| `ROBOFLOW_PROJECT` | `uterine-fibroid-oswew-wphbc` | Project identifier |
| `ROBOFLOW_VERSION` | `1` | Dataset version |
| `DATASET_FORMAT` | `yolov11` | Bounding box annotation format |
| `MINIO_ENDPOINT` | `http://minio-ml-workshop:9000` | In-cluster MinIO API URL |
| `MINIO_BUCKET` | `fibroid-data` | Target S3 bucket name |

---

## 🧪 Local Testing

You can run the ingestion script standalone at any time:

```bash
pip install -r requirements.txt
python scripts/fetch_and_archive_data.py
```
