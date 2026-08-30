from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINED_MODEL = PROJECT_ROOT / "models" / "best.pt"
BASE_MODEL = PROJECT_ROOT / "yolo11n.pt"

MODEL_PATH = (
    TRAINED_MODEL
    if TRAINED_MODEL.exists()
    else BASE_MODEL
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DETECTOR
# ============================================================

class ThreatDetector:

    def __init__(
        self,
        confidence: float = 0.35,
    ) -> None:

        self.confidence = confidence
        self.device = DEVICE

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        print(
            f"[MODEL] Loading: {MODEL_PATH}"
        )

        self.model = YOLO(
            str(MODEL_PATH)
        )

        model_names = self.model.names

        if isinstance(
            model_names,
            dict,
        ):
            self.names = model_names
        else:
            self.names = dict(
                enumerate(
                    model_names
                )
            )

        print(
            f"[MODEL] Classes: "
            f"{len(self.names)}"
        )

        print(
            f"[MODEL] Device: "
            f"{self.device_name}"
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    @property
    def model_name(self) -> str:
        return MODEL_PATH.name

    @property
    def class_count(self) -> int:
        return len(self.names)

    @property
    def device_name(self) -> str:

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(
                0
            )

        return "CPU"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    def set_confidence(
        self,
        value: float,
    ) -> None:

        self.confidence = max(
            0.01,
            min(
                0.99,
                float(value),
            ),
        )

    # ========================================================
    # CLASS NAME
    # ========================================================

    def get_class_name(
        self,
        class_id: int,
    ) -> str:

        return str(
            self.names.get(
                class_id,
                f"class_{class_id}",
            )
        )

    # ========================================================
    # PERSON
    # ========================================================

    @staticmethod
    def is_person_class(
        class_name: str,
    ) -> bool:

        normalized = (
            class_name
            .strip()
            .lower()
        )

        return normalized in {
            "person",
            "normal_person",
            "potentially_dangerous_person",
        }

    # ========================================================
    # THREAT
    # ========================================================

    @staticmethod
    def is_threat_class(
        class_name: str,
    ) -> bool:

        normalized = (
            class_name
            .strip()
            .lower()
        )

        return (
            normalized
            == "potentially_dangerous_person"
        )

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        # ----------------------------------------------------
        # YOLO TRACK
        # ----------------------------------------------------

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            imgsz=480,
            verbose=False,
        )

        result = results[0]

        persons = 0
        threats = 0
        highest_confidence = 0.0

        detections: list[
            dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            track_ids = (
                result.boxes.id
            )

            for index, box in enumerate(
                result.boxes
            ):

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                class_name = (
                    self.get_class_name(
                        class_id
                    )
                )

                is_person = (
                    self.is_person_class(
                        class_name
                    )
                )

                is_threat = (
                    self.is_threat_class(
                        class_name
                    )
                )

                if is_person:
                    persons += 1

                if is_threat:
                    threats += 1

                highest_confidence = max(
                    highest_confidence,
                    confidence,
                )

                # --------------------------------------------
                # TRACK ID
                # --------------------------------------------

                track_id = None

                if track_ids is not None:
                    track_id = int(
                        track_ids[index]
                    )

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = [
                    float(value)
                    for value in
                    box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                        "is_person": is_person,
                        "is_threat": is_threat,
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ],
                    }
                )

        # ----------------------------------------------------
        # DRAW BASE FRAME
        # ----------------------------------------------------

        annotated = frame.copy()

        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = map(
                int,
                detection["bbox"],
            )

            track_id = (
                detection["track_id"]
            )

            confidence = (
                detection["confidence"]
            )

            class_name = (
                detection["class_name"]
            )

            is_person = (
                detection["is_person"]
            )

            is_threat = (
                detection["is_threat"]
            )

            # ----------------------------------------------
            # LABEL
            # ----------------------------------------------

            if is_threat:

                if track_id is not None:
                    label = (
                        f"THREAT "
                        f"T-{track_id:02d} "
                        f"{confidence:.2f}"
                    )
                else:
                    label = (
                        f"THREAT "
                        f"{confidence:.2f}"
                    )

                # OpenCV BGR
                box_color = (
                    70,
                    80,
                    220,
                )

            elif is_person:

                if track_id is not None:
                    label = (
                        f"PERSON "
                        f"P-{track_id:02d} "
                        f"{confidence:.2f}"
                    )
                else:
                    label = (
                        f"PERSON "
                        f"{confidence:.2f}"
                    )

                box_color = (
                    70,
                    145,
                    225,
                )

            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                box_color = (
                    130,
                    130,
                    130,
                )

            # ----------------------------------------------
            # BOX
            # ----------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            # ----------------------------------------------
            # LABEL SIZE
            # ----------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.50
            thickness = 1

            (
                text_width,
                text_height,
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            label_bottom = max(
                y1,
                text_height + 10,
            )

            label_top = (
                label_bottom
                - text_height
                - 10
            )

            # ----------------------------------------------
            # LABEL BACKGROUND
            # ----------------------------------------------

            cv2.rectangle(
                annotated,
                (
                    x1,
                    label_top,
                ),
                (
                    x1 +
                    text_width +
                    8,
                    label_bottom,
                ),
                box_color,
                -1,
            )

            # ----------------------------------------------
            # TEXT
            # ----------------------------------------------

            cv2.putText(
                annotated,
                label,
                (
                    x1 + 4,
                    label_bottom - 5,
                ),
                font,
                font_scale,
                (
                    255,
                    255,
                    255,
                ),
                thickness,
                cv2.LINE_AA,
            )

        # ====================================================
        # TIMING
        # ====================================================

        processing_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return {
            "frame": annotated,
            "persons": persons,
            "threats": threats,
            "objects": len(
                detections
            ),
            "highest_confidence": round(
                highest_confidence,
                4,
            ),
            "processing_ms": round(
                processing_ms,
                2,
            ),
            "detections": detections,
            "model": self.model_name,
            "device": (
                "CUDA"
                if torch.cuda.is_available()
                else "CPU"
            ),
        }


# ============================================================
# GLOBAL DETECTOR
# ============================================================

detector = ThreatDetector()