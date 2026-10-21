import os
import uuid
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# Ensure legacy Keras, disable oneDNN conflicts, and suppress C++ warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
import easyocr


class ANPREngine:
    """
    Automatic Number Plate Recognition (ANPR) Engine
    Encapsulates SSD MobileNet V2 detection and EasyOCR recognition.
    """

    def __init__(
        self,
        config_path: str = "models/my_ssd_mobnet/pipeline.config",
        checkpoint_dir: str = "models/my_ssd_mobnet",
        checkpoint_name: str = "ckpt-11",
        output_folder: str = "Detection_Images",
        csv_path: str = "detection_results.csv",
        use_gpu: bool = False
    ):
        self.config_path = config_path
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.output_folder = output_folder
        self.csv_path = csv_path
        self.use_gpu = use_gpu

        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize detector
        self.detection_model = self._load_detector()

        # Initialize OCR reader
        self.reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)

    def _load_detector(self):
        configs = config_util.get_configs_from_pipeline_file(self.config_path)
        detection_model = model_builder.build(
            model_config=configs["model"], is_training=False
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.checkpoint_path).expect_partial()
        return detection_model

    @tf.function
    def _detect_fn(self, image_tensor):
        image, shapes = self.detection_model.preprocess(image_tensor)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect(
        self,
        image_bgr: np.ndarray,
        detection_threshold: float = 0.5,
        region_threshold: float = 0.25,
        ocr_confidence_threshold: float = 0.2
    ) -> Dict:
        """
        Runs license plate detection and text extraction on a BGR image.
        Returns a dictionary containing detection metadata, annotated image, and plate details.
        """
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_bgr, 0), dtype=tf.float32)

        raw_detections = self._detect_fn(input_tensor)

        num_detections = int(raw_detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in raw_detections.items()
        }
        detections["num_detections"] = num_detections

        scores = detections["detection_scores"]
        boxes = detections["detection_boxes"]

        detected_plates = []
        annotated_image = image_bgr.copy()

        for idx, (score, box) in enumerate(zip(scores, boxes)):
            if score < detection_threshold:
                continue

            ymin, xmin, ymax, xmax = box
            ymin_px = max(0, int(ymin * orig_h))
            xmin_px = max(0, int(xmin * orig_w))
            ymax_px = min(orig_h, int(ymax * orig_h))
            xmax_px = min(orig_w, int(xmax * orig_w))

            if ymax_px <= ymin_px or xmax_px <= xmin_px:
                continue

            # Crop Region of Interest (ROI)
            plate_crop = image_bgr[ymin_px:ymax_px, xmin_px:xmax_px]
            if plate_crop.size == 0:
                continue

            # Run OCR on cropped plate
            ocr_results = self.reader.readtext(plate_crop)

            # Filter and parse extracted text
            plate_text, ocr_conf = self._parse_plate_text(
                plate_crop, ocr_results, region_threshold, ocr_confidence_threshold
            )

            detected_plates.append({
                "plate_text": plate_text,
                "detection_score": float(score),
                "ocr_confidence": float(ocr_conf),
                "box": (ymin_px, xmin_px, ymax_px, xmax_px),
                "plate_crop": plate_crop,
                "raw_ocr": ocr_results
            })

            # Draw visual markers on annotated image
            self._draw_detection(
                annotated_image,
                (ymin_px, xmin_px, ymax_px, xmax_px),
                plate_text,
                score
            )

        return {
            "annotated_image": annotated_image,
            "plates": detected_plates,
            "count": len(detected_plates)
        }

    def _parse_plate_text(
        self,
        region: np.ndarray,
        ocr_results: List,
        region_threshold: float,
        ocr_confidence_threshold: float
    ) -> Tuple[str, float]:
        """
        Filters text by size ratio and confidence.
        """
        if not ocr_results:
            return "NO TEXT DETECTED", 0.0

        rectangle_size = region.shape[0] * region.shape[1]
        valid_texts = []
        confidences = []

        for result in ocr_results:
            bbox, text, conf = result
            if conf < ocr_confidence_threshold:
                continue

            # Calculate box area
            p1 = np.array(bbox[0])
            p2 = np.array(bbox[1])
            p3 = np.array(bbox[2])

            width = np.linalg.norm(p2 - p1)
            height = np.linalg.norm(p3 - p2)
            token_area = width * height

            # Relative area check to filter noise
            if rectangle_size > 0 and (token_area / rectangle_size) >= region_threshold:
                cleaned_text = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
                if cleaned_text:
                    valid_texts.append(cleaned_text.upper())
                    confidences.append(float(conf))

        if not valid_texts:
            # Fallback to the highest confidence text result if size filtering was too strict
            best_res = max(ocr_results, key=lambda x: x[2])
            cleaned = "".join(c for c in best_res[1] if c.isalnum() or c.isspace()).strip()
            return cleaned.upper() if cleaned else "UNKNOWN", float(best_res[2])

        final_plate_text = " ".join(valid_texts)
        avg_confidence = float(np.mean(confidences))
        return final_plate_text, avg_confidence

    def _draw_detection(
        self,
        img: np.ndarray,
        box: Tuple[int, int, int, int],
        text: str,
        score: float
    ):
        ymin, xmin, ymax, xmax = box
        # Box styling
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 220, 255), 3)

        label = f"{text} ({score * 100:.1f}%)"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness = 2
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_ymin = max(ymin - 10, h + 10)
        cv2.rectangle(
            img,
            (xmin, label_ymin - h - 8),
            (xmin + w + 10, label_ymin + baseline - 4),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            img,
            (xmin, label_ymin - h - 8),
            (xmin + w + 10, label_ymin + baseline - 4),
            (0, 220, 255),
            1
        )
        cv2.putText(
            img,
            label,
            (xmin + 5, label_ymin - 4),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    def save_result(self, plate_crop: np.ndarray, plate_text: str) -> str:
        """
        Saves cropped plate image and appends record to CSV.
        Returns the saved image filename.
        """
        img_name = f"{uuid.uuid1()}.jpg"
        save_path = os.path.join(self.output_folder, img_name)
        cv2.imwrite(save_path, plate_crop)

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([img_name, plate_text, datetime.now().isoformat()])

        return img_name
