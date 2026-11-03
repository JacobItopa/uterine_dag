import os
import io
import time
import base64
import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UterineFibroidPredictor")


class Predictor(object):
    """
    Seldon Core Predictor component for real-time Uterine Fibroid Detection.
    Optimized for live ultrasound scan frame processing via REST & gRPC.
    """
    def __init__(self):
        self.weights_path = os.getenv("MODEL_WEIGHTS_PATH", "best.pt")
        logger.info(f"Loading YOLOv11 model weights from: {self.weights_path}...")
        self.model = YOLO(self.weights_path)
        logger.info(f"Model successfully loaded. Active classes: {self.model.names}")

    def class_names(self):
        """Returns feature/output column names for Seldon response."""
        return ["class_id", "label", "confidence", "x1", "y1", "x2", "y2"]

    def _decode_image(self, item):
        """Decodes an image from base64 string, bytearray, or numpy array."""
        if isinstance(item, str):
            # Strip data URI header if present (e.g. 'data:image/jpeg;base64,...')
            if "," in item:
                item = item.split(",", 1)[1]
            img_bytes = base64.b64decode(item)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(item, (bytes, bytearray)):
            return Image.open(io.BytesIO(item)).convert("RGB")
        elif isinstance(item, np.ndarray):
            return item
        else:
            raise ValueError(f"Unsupported image frame type: {type(item)}")

    def predict(self, data_array, feature_names=None, meta=None, **kwargs):
        """
        Processes incoming ultrasound scan frames and returns detected bounding boxes.
        
        Input:
            data_array: List or ndarray containing base64 string(s) of image frames.
        Output:
            ndarray of detections: [[class_id, class_name, confidence, x1, y1, x2, y2], ...]
        """
        start_time = time.time()
        detections_list = []

        # Extract frames to process
        if isinstance(data_array, dict):
            # Handle custom JSON payload e.g. {"image": "..."}
            frames = [data_array.get("image", "")]
            conf_threshold = float(data_array.get("conf", 0.25))
        elif isinstance(data_array, (list, np.ndarray)):
            frames = list(data_array)
            conf_threshold = 0.25
        else:
            frames = [data_array]
            conf_threshold = 0.25

        for frame_item in frames:
            try:
                img = self._decode_image(frame_item)
                results = self.model.predict(img, conf=conf_threshold, verbose=False)

                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        for b in boxes:
                            xyxy = b.xyxy[0].tolist()
                            conf = float(b.conf[0])
                            cls_id = int(b.cls[0])
                            cls_name = self.model.names.get(cls_id, str(cls_id))

                            detections_list.append([
                                cls_id,
                                cls_name,
                                round(conf, 4),
                                round(xyxy[0], 2),
                                round(xyxy[1], 2),
                                round(xyxy[2], 2),
                                round(xyxy[3], 2)
                            ])
            except Exception as e:
                logger.error(f"Error during frame inference: {e}")

        # If no detections found in frame, return empty indicator
        if not detections_list:
            detections_list = [[-1, "none", 0.0, 0.0, 0.0, 0.0, 0.0]]

        inference_time_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Inference completed in {inference_time_ms} ms. Detections count: {len(detections_list)}")

        return np.array(detections_list, dtype=object)
