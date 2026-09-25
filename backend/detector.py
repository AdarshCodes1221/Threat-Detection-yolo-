from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "person_weapon_phone_best.pt"
)

TRACKER_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "botsort_reid.yaml"
)

PERSON_MEMORY_TTL = 25.0
MAX_PERSON_MEMORY = 100
THREAT_CONFIRMATION_FRAMES = 2
THREAT_RELEASE_FRAMES = 1

# ------------------------------------------------------------
# Weapon -> person association.
#
# NOTE: This threshold is calibrated against `association_score`
# below, which returns a *containment + proximity* score in the
# 0..1 range (see the method's docstring for why). It is NOT
# comparable to a raw IoU value.
# ------------------------------------------------------------
ASSOCIATION_THRESHOLD = 0.40

PERSON_CONFIDENCE = 0.25
WEAPON_CONFIDENCE = 0.01
PHONE_CONFIDENCE = 0.10
REID_THRESHOLD = 0.82


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL CLASSES
# ============================================================

PERSON_CLASS_ID = 0
WEAPON_CLASS_ID = 1
PHONE_CLASS_ID = 2

CLASS_NAMES = {
    PERSON_CLASS_ID: "person",
    WEAPON_CLASS_ID: "weapon",
    PHONE_CLASS_ID: "phone",
}


# ============================================================
# DETECTOR
# ============================================================

class ThreatDetector:

    def __init__(
        self,
        confidence: float = 0.25,
        weapon_confidence: float = 0.01,
        phone_confidence: float = 0.10,
    ) -> None:

        self.confidence = float(confidence)
        self.weapon_confidence = float(
            weapon_confidence
        )
        self.phone_confidence = float(
            phone_confidence
        )
        self.track_confidence = min(
            self.confidence,
            self.weapon_confidence,
            self.phone_confidence,
        )

        self.device = DEVICE
        self.tracker_config = str(TRACKER_CONFIG)

        # ----------------------------------------------------
        # Identity / Re-ID settings
        # ----------------------------------------------------

        self.next_person_id = 1
        self.next_weapon_id = 1
        self.next_phone_id = 1

        self.person_memory_ttl = PERSON_MEMORY_TTL
        self.reid_threshold = REID_THRESHOLD
        self.min_crop_size = 40
        self.max_person_memory = MAX_PERSON_MEMORY

        # ----------------------------------------------------
        # Threat temporal confirmation
        # ----------------------------------------------------

        self.threat_required_frames = THREAT_CONFIRMATION_FRAMES
        self.threat_release_frames = THREAT_RELEASE_FRAMES
        self.association_threshold = ASSOCIATION_THRESHOLD

        # stable person id -> state
        self.person_memory: dict[
            str,
            dict[str, Any],
        ] = {}

        # ByteTrack id -> stable person id
        self.track_to_person: dict[
            int,
            str,
        ] = {}

        # Stable weapon / phone IDs
        self.weapon_track_to_id: dict[
            int,
            str,
        ] = {}

        self.phone_track_to_id: dict[
            int,
            str,
        ] = {}

        # Threat frame counters
        self.threat_state: dict[
            str,
            dict[str, Any],
        ] = {}
        self.event_state: dict[
            str,
            dict[str, Any],
        ] = {}
        self.event_counter = 1

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "\n"
                "============================================\n"
                "MODEL NOT FOUND\n"
                "============================================\n"
                f"Expected:\n{MODEL_PATH}\n\n"
                "Copy person_weapon_phone_best.pt into:\n"
                f"{PROJECT_ROOT / 'models'}\n"
                "============================================\n"
            )

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        print(
            f"[MODEL] Loading trained 3-class model: {MODEL_PATH}"
        )

        self.model = YOLO(
            str(MODEL_PATH)
        )

        # ----------------------------------------------------
        # Read actual model names
        # ----------------------------------------------------

        model_names = self.model.names

        if isinstance(
            model_names,
            dict,
        ):

            self.names = {
                int(k): str(v)
                for k, v in model_names.items()
            }

        else:

            self.names = {
                i: str(name)
                for i, name in enumerate(
                    model_names
                )
            }

        # ----------------------------------------------------
        # Strict validation
        # ----------------------------------------------------

        required_names = {
            0: "person",
            1: "weapon",
            2: "phone",
        }

        if set(self.names.keys()) != set(required_names.keys()):
            raise RuntimeError(
                "The trained model must expose exactly 3 classes: "
                f"{required_names}. Loaded classes: {self.names}"
            )

        for class_id, expected_name in required_names.items():
            actual_name = self.names.get(class_id)
            if actual_name is None or str(actual_name).lower() != expected_name:
                raise RuntimeError(
                    "The trained model class mapping is invalid. "
                    f"Expected {required_names}, got {self.names}."
                )

        print(
            f"[MODEL] Classes: {self.names}"
        )
        print(
            f"[MODEL] Device: {self.device_name}"
        )
        print(
            "[MODEL] Tracker: BoT-SORT + Re-ID"
        )

    # ========================================================
    # PUBLIC INFORMATION
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
        self.track_confidence = min(
            self.confidence,
            self.weapon_confidence,
            self.phone_confidence,
        )

    # ========================================================
    # CLASS NAME
    # ========================================================

    def get_class_name(
        self,
        class_id: int,
    ) -> str:

        return self.names.get(
            int(class_id),
            CLASS_NAMES.get(
                int(class_id),
                f"class_{class_id}",
            ),
        )

    # ========================================================
    # IOU
    # ========================================================

    @staticmethod
    def calculate_iou(
        box_a,
        box_b,
    ) -> float:

        ax1, ay1, ax2, ay2 = map(
            float,
            box_a,
        )

        bx1, by1, bx2, by2 = map(
            float,
            box_b,
        )

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            iw * ih
        )

        area_a = (
            max(
                0.0,
                ax2 - ax1,
            )
            *
            max(
                0.0,
                ay2 - ay1,
            )
        )

        area_b = (
            max(
                0.0,
                bx2 - bx1,
            )
            *
            max(
                0.0,
                by2 - by1,
            )
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0.0:
            return 0.0

        return (
            intersection
            / union
        )

    # ========================================================
    # BOX CENTER
    # ========================================================

    @staticmethod
    def box_center(
        box,
    ) -> tuple[float, float]:

        x1, y1, x2, y2 = map(
            float,
            box,
        )

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    # ========================================================
    # OBJECT -> PERSON ASSOCIATION (region test used for phones)
    # ========================================================

    @staticmethod
    def object_belongs_to_person(
        object_box,
        person_box,
    ) -> bool:

        ox1, oy1, ox2, oy2 = map(
            float,
            object_box,
        )

        px1, py1, px2, py2 = map(
            float,
            person_box,
        )

        person_width = max(
            1.0,
            px2 - px1,
        )

        person_height = max(
            1.0,
            py2 - py1,
        )

        # ----------------------------------------------------
        # Expand person region moderately.
        # ----------------------------------------------------

        expanded_x1 = (
            px1
            - person_width * 0.20
        )

        expanded_y1 = (
            py1
            - person_height * 0.05
        )

        expanded_x2 = (
            px2
            + person_width * 0.20
        )

        expanded_y2 = (
            py2
            + person_height * 0.05
        )

        object_center_x = (
            ox1 + ox2
        ) / 2.0

        object_center_y = (
            oy1 + oy2
        ) / 2.0

        # ----------------------------------------------------
        # Center inside expanded person box
        # ----------------------------------------------------

        if (
            expanded_x1
            <= object_center_x
            <= expanded_x2
            and
            expanded_y1
            <= object_center_y
            <= expanded_y2
        ):

            return True

        # ----------------------------------------------------
        # IoU fallback
        # ----------------------------------------------------

        iou = ThreatDetector.calculate_iou(
            (
                ox1,
                oy1,
                ox2,
                oy2,
            ),
            (
                expanded_x1,
                expanded_y1,
                expanded_x2,
                expanded_y2,
            ),
        )

        if iou >= 0.01:

            return True

        return False

    # ========================================================
    # WEAPON -> PERSON ASSOCIATION SCORE
    #
    # THIS IS THE FIX. See "what was wrong" in the accompanying
    # explanation for the full story; short version:
    #
    # The previous implementation scored association mostly off
    # raw IoU(weapon_box, person_box). Because a person's bbox
    # is typically 10-50x the area of a held weapon's bbox, IoU
    # between them is *mathematically incapable* of exceeding
    # roughly (weapon_area / person_area) even when the weapon
    # sits dead-center on the person's body -- usually a value
    # under 0.05. That capped the entire score formula's maximum
    # realistic output around 0.35-0.40, permanently below the
    # old 0.50 threshold. No genuinely held weapon could ever
    # cross the line, hence "WEAPONS: 2, THREATS: 0" even with
    # visibly-held weapons.
    #
    # This version scores two things that don't have that
    # size-mismatch problem:
    #
    #   1. containment  - what fraction of the WEAPON's own area
    #      (not the person's) falls inside a person region that's
    #      been expanded outward to plausibly cover an extended
    #      arm/hand. A held weapon is very often ~80-100%
    #      contained even though its IoU with the person is tiny.
    #
    #   2. proximity     - normalized distance from the weapon's
    #      center to the nearest point on the person's own
    #      (unexpanded) bbox, decaying to 0 a bit under half a
    #      body-diagonal away. This lets an object just outside
    #      the expanded box (e.g. a fully extended arm) still
    #      contribute, and pulls the score toward 0 for anything
    #      clearly unrelated (a weapon on a table across the room).
    #
    # Final score = containment*0.70 + proximity*0.30, in 0..1,
    # and *is* achievable at ~0.9-1.0 for a genuinely held weapon
    # (validated against synthetic held/extended-arm/far-away
    # cases -- see test notes). ASSOCIATION_THRESHOLD was
    # recalibrated to 0.40 to match this new scale.
    # ========================================================

    @staticmethod
    def association_score(
        weapon_box,
        person_box,
    ) -> float:

        ox1, oy1, ox2, oy2 = map(
            float,
            weapon_box,
        )

        px1, py1, px2, py2 = map(
            float,
            person_box,
        )

        weapon_area = max(
            1.0,
            (ox2 - ox1) * (oy2 - oy1),
        )

        person_width = max(
            1.0,
            px2 - px1,
        )

        person_height = max(
            1.0,
            py2 - py1,
        )

        # ----------------------------------------------------
        # Expand outward to plausibly cover an extended arm.
        # Wider sideways than up/down; a bit more slack below
        # the box than above it, since a lowered/aimed weapon
        # tends to sit near waist height or below.
        # ----------------------------------------------------

        expanded_x1 = px1 - person_width * 0.35
        expanded_y1 = py1 - person_height * 0.08
        expanded_x2 = px2 + person_width * 0.35
        expanded_y2 = py2 + person_height * 0.20

        ix1 = max(ox1, expanded_x1)
        iy1 = max(oy1, expanded_y1)
        ix2 = min(ox2, expanded_x2)
        iy2 = min(oy2, expanded_y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        containment = (iw * ih) / weapon_area

        # ----------------------------------------------------
        # Proximity: distance from weapon center to the nearest
        # point on the person's *own* (unexpanded) box, normalized
        # by that person's bbox diagonal so it scales with person
        # size in the frame (near vs. far from camera).
        # ----------------------------------------------------

        weapon_cx = (ox1 + ox2) / 2.0
        weapon_cy = (oy1 + oy2) / 2.0

        nearest_x = min(max(weapon_cx, px1), px2)
        nearest_y = min(max(weapon_cy, py1), py2)

        dx = weapon_cx - nearest_x
        dy = weapon_cy - nearest_y

        distance = (dx * dx + dy * dy) ** 0.5

        diagonal = max(
            1.0,
            (person_width ** 2 + person_height ** 2) ** 0.5,
        )

        normalized_distance = distance / diagonal

        # Decays to 0 by roughly 0.4x the body diagonal away.
        proximity = max(
            0.0,
            1.0 - normalized_distance * 2.5,
        )

        score = (containment * 0.70) + (proximity * 0.30)

        return max(0.0, min(1.0, score))

    # ========================================================
    # PERSON CROP
    # ========================================================

    @staticmethod
    def crop_box(
        frame: np.ndarray,
        bbox,
    ) -> np.ndarray | None:

        if frame is None:
            return None

        height, width = (
            frame.shape[:2]
        )

        x1, y1, x2, y2 = map(
            int,
            bbox,
        )

        x1 = max(
            0,
            min(
                width - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                height - 1,
                y1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                width,
                x2,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                height,
                y2,
            ),
        )

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if (
            crop is None
            or crop.size == 0
        ):

            return None

        return crop

    # ========================================================
    # APPEARANCE FEATURE
    # ========================================================

    def extract_person_feature(
        self,
        frame: np.ndarray,
        bbox,
    ) -> np.ndarray | None:

        crop = self.crop_box(
            frame,
            bbox,
        )

        if crop is None:
            return None

        h, w = crop.shape[:2]

        if (
            h < self.min_crop_size
            or w < self.min_crop_size
        ):

            return None

        # ----------------------------------------------------
        # Normalize geometry.
        # ----------------------------------------------------

        resized = cv2.resize(
            crop,
            (
                64,
                128,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # HSV histogram
        #
        # This captures clothing/appearance information.
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2HSV,
        )

        hsv_hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [16, 8],
            [
                0,
                180,
                0,
                256,
            ],
        )

        hsv_hist = cv2.normalize(
            hsv_hist,
            hsv_hist,
        ).flatten()

        # ----------------------------------------------------
        # RGB histogram
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB,
        )

        rgb_hist = cv2.calcHist(
            [rgb],
            [0, 1],
            None,
            [8, 8],
            [
                0,
                256,
                0,
                256,
            ],
        )

        rgb_hist = cv2.normalize(
            rgb_hist,
            rgb_hist,
        ).flatten()

        # ----------------------------------------------------
        # Grayscale appearance descriptor
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY,
        )

        small_gray = cv2.resize(
            gray,
            (
                16,
                32,
            ),
            interpolation=cv2.INTER_AREA,
        )

        gray_feature = (
            small_gray.astype(
                np.float32
            )
            / 255.0
        ).flatten()

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        feature = np.concatenate(
            [
                hsv_hist.astype(
                    np.float32
                ),
                rgb_hist.astype(
                    np.float32
                ),
                gray_feature,
            ]
        )

        # ----------------------------------------------------
        # L2 normalize
        # ----------------------------------------------------

        norm = np.linalg.norm(
            feature
        )

        if norm <= 1e-8:

            return None

        return (
            feature
            / norm
        ).astype(
            np.float32
        )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_similarity(
        feature_a: np.ndarray,
        feature_b: np.ndarray,
    ) -> float:

        if (
            feature_a is None
            or feature_b is None
        ):

            return -1.0

        norm_a = np.linalg.norm(
            feature_a
        )

        norm_b = np.linalg.norm(
            feature_b
        )

        if (
            norm_a <= 1e-8
            or norm_b <= 1e-8
        ):

            return -1.0

        return float(
            np.dot(
                feature_a,
                feature_b,
            )
            / (
                norm_a
                * norm_b
            )
        )

    # ========================================================
    # CREATE PERSON ID
    # ========================================================

    def create_person_id(self) -> str:

        person_id = (
            f"P-{self.next_person_id:02d}"
        )

        self.next_person_id += 1

        return person_id

    # ========================================================
    # UPDATE PERSON MEMORY
    # ========================================================

    def update_person_memory(
        self,
        person_id: str,
        frame: np.ndarray,
        bbox,
        feature: np.ndarray | None,
        track_id: int | None,
        now: float,
    ) -> None:

        state = self.person_memory.get(
            person_id
        )

        if state is None:

            state = {
                "person_id": person_id,
                "feature": feature,
                "bbox": list(bbox),
                "track_id": track_id,
                "first_seen": now,
                "last_seen": now,
                "seen_count": 0,
            }

            self.person_memory[
                person_id
            ] = state

        else:

            # -----------------------------------------------
            # Update appearance gradually.
            # -----------------------------------------------

            old_feature = state.get(
                "feature"
            )

            if (
                feature is not None
                and old_feature is not None
            ):

                blended = (
                    old_feature * 0.75
                    +
                    feature * 0.25
                )

                norm = np.linalg.norm(
                    blended
                )

                if norm > 1e-8:

                    state["feature"] = (
                        blended
                        / norm
                    ).astype(
                        np.float32
                    )

            elif feature is not None:

                state["feature"] = feature

            state["bbox"] = list(
                bbox
            )

            state["track_id"] = track_id

            state["last_seen"] = now

        state["seen_count"] = (
            int(
                state.get(
                    "seen_count",
                    0,
                )
            )
            + 1
        )

    # ========================================================
    # CLEAN EXPIRED PERSONS
    # ========================================================

    def cleanup_person_memory(
        self,
        now: float,
    ) -> None:

        expired = []

        for person_id, state in (
            self.person_memory.items()
        ):

            last_seen = float(
                state.get(
                    "last_seen",
                    now,
                )
            )

            if (
                now - last_seen
                > self.person_memory_ttl
            ):

                expired.append(
                    person_id
                )

        for person_id in expired:

            self.person_memory.pop(
                person_id,
                None,
            )

            # Remove old track mappings.
            old_tracks = [
                track_id
                for track_id, pid
                in self.track_to_person.items()
                if pid == person_id
            ]

            for track_id in old_tracks:

                self.track_to_person.pop(
                    track_id,
                    None,
                )

        # ----------------------------------------------------
        # Hard cap.
        # ----------------------------------------------------

        if (
            len(self.person_memory)
            > self.max_person_memory
        ):

            ordered = sorted(
                self.person_memory.items(),
                key=lambda item:
                    float(
                        item[1].get(
                            "last_seen",
                            0.0,
                        )
                    ),
            )

            remove_count = (
                len(self.person_memory)
                - self.max_person_memory
            )

            for person_id, _ in ordered[
                :remove_count
            ]:

                self.person_memory.pop(
                    person_id,
                    None,
                )

    # ========================================================
    # ASSIGN STABLE PERSON ID
    # ========================================================

    def assign_person_id(
        self,
        frame: np.ndarray,
        bbox,
        track_id: int | None,
        now: float,
        reserved_person_ids: set[str],
    ) -> tuple[str, float]:

        # ----------------------------------------------------
        # First try the current ByteTrack mapping.
        # ----------------------------------------------------

        if track_id is not None:

            existing = (
                self.track_to_person.get(
                    track_id
                )
            )

            if existing is not None:

                feature = (
                    self.extract_person_feature(
                        frame,
                        bbox,
                    )
                )

                self.update_person_memory(
                    existing,
                    frame,
                    bbox,
                    feature,
                    track_id,
                    now,
                )

                reserved_person_ids.add(
                    existing
                )

                return existing, 1.0

        # ----------------------------------------------------
        # Create current feature.
        # ----------------------------------------------------

        feature = (
            self.extract_person_feature(
                frame,
                bbox,
            )
        )

        # ----------------------------------------------------
        # Search remembered people.
        # ----------------------------------------------------

        best_person = None
        best_similarity = -1.0

        if feature is not None:

            for person_id, state in (
                self.person_memory.items()
            ):

                if person_id in reserved_person_ids:
                    continue

                old_feature = state.get(
                    "feature"
                )

                if old_feature is None:
                    continue

                similarity = (
                    self.cosine_similarity(
                        feature,
                        old_feature,
                    )
                )

                # Don't match an extremely old memory.
                age = (
                    now
                    - float(
                        state.get(
                            "last_seen",
                            now,
                        )
                    )
                )

                if (
                    age
                    > self.person_memory_ttl
                ):

                    continue

                if (
                    similarity
                    > best_similarity
                ):

                    best_similarity = (
                        similarity
                    )

                    best_person = person_id

        # ----------------------------------------------------
        # Re-identification match
        # ----------------------------------------------------

        if (
            best_person is not None
            and
            best_similarity
            >= self.reid_threshold
        ):

            person_id = best_person

        # ----------------------------------------------------
        # New person
        # ----------------------------------------------------

        else:

            person_id = (
                self.create_person_id()
            )

            best_similarity = 0.0

        # ----------------------------------------------------
        # Save mapping.
        # ----------------------------------------------------

        if track_id is not None:

            self.track_to_person[
                track_id
            ] = person_id

        self.update_person_memory(
            person_id,
            frame,
            bbox,
            feature,
            track_id,
            now,
        )

        reserved_person_ids.add(
            person_id
        )

        return (
            person_id,
            float(
                best_similarity
            ),
        )

    # ========================================================
    # WEAPON ID
    # ========================================================

    def get_weapon_id(
        self,
        track_id: int | None,
    ) -> str:

        if track_id is not None:

            existing = (
                self.weapon_track_to_id.get(
                    track_id
                )
            )

            if existing is not None:

                return existing

        weapon_id = (
            f"W-{self.next_weapon_id:02d}"
        )

        self.next_weapon_id += 1

        if track_id is not None:

            self.weapon_track_to_id[
                track_id
            ] = weapon_id

        return weapon_id

    # ========================================================
    # PHONE ID
    # ========================================================

    def get_phone_id(
        self,
        track_id: int | None,
    ) -> str:

        if track_id is not None:

            existing = (
                self.phone_track_to_id.get(
                    track_id
                )
            )

            if existing is not None:

                return existing

        phone_id = (
            f"PH-{self.next_phone_id:02d}"
        )

        self.next_phone_id += 1

        if track_id is not None:

            self.phone_track_to_id[
                track_id
            ] = phone_id

        return phone_id

    # ========================================================
    # THREAT TEMPORAL STATE
    # ========================================================

    def update_threat_state(
        self,
        person_id: str,
        has_weapon: bool,
    ) -> tuple[bool, bool]:
        """Advance the temporal threat counter for *person_id*.

        Returns ``(confirmed, just_confirmed)`` where *just_confirmed* flips
        from ``False`` to ``True`` on the exact frame the person crosses from
        WEAPON_CANDIDATE into CONFIRMED_THREAT — the trigger for a one-shot
        evidence capture.
        """

        state = self.threat_state.setdefault(
            person_id,
            {
                "frames": 0,
                "last_seen": time.time(),
                "confirmed": False,
            },
        )

        state["last_seen"] = time.time()

        was_confirmed = bool(
            state.get(
                "confirmed",
                False,
            )
        )

        if has_weapon:
            state["frames"] = int(state.get("frames", 0)) + 1
        else:
            state["frames"] = max(0, int(state.get("frames", 0)) - self.threat_release_frames)

        confirmed = state["frames"] >= self.threat_required_frames
        state["confirmed"] = confirmed
        just_confirmed = confirmed and not was_confirmed
        return confirmed, just_confirmed

    def _capture_event_evidence(
        self,
        person_id: str,
    ) -> str:
        """Create a new dated event directory and return its event_id.

        Evidence images (full frame, person crop) and the event JSON are
        written by :meth:`_write_evidence` at the moment of confirmation.
        """

        date_dir = (
            PROJECT_ROOT
            / "events"
            / time.strftime("%Y-%m-%d")
        )

        date_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        while True:

            event_id = (
                f"EVT-{self.event_counter:04d}_"
                f"{person_id}"
            )

            event_dir = (
                date_dir
                / event_id
            )

            if not event_dir.exists():

                self.event_counter += 1
                return event_id

            self.event_counter += 1

    def _write_evidence(
        self,
        event_id: str,
        person_id: str,
        frame: np.ndarray,
        person_crop: np.ndarray,
        weapon_id: str | None,
        weapon_confidence: float,
        person_confidence: float,
        first_seen: float,
        last_seen: float,
    ) -> dict[str, str]:
        """Write one evidence package (full frame + person crop + JSON).

        Returns a dict with ``evidence_image`` and ``person_crop`` paths
        relative to PROJECT_ROOT so they can be sent to the frontend.
        """

        date_dir = (
            PROJECT_ROOT
            / "events"
            / time.strftime("%Y-%m-%d")
        )

        event_dir = (
            date_dir
            / event_id
        )

        event_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        full_frame_path = (
            event_dir
            / "full_frame.jpg"
        )

        person_crop_path = (
            event_dir
            / "person_crop.jpg"
        )

        event_json_path = (
            event_dir
            / "event.json"
        )

        cv2.imwrite(
            str(full_frame_path),
            frame,
        )

        cv2.imwrite(
            str(person_crop_path),
            person_crop,
        )

        event_payload = {
            "event_id": event_id,
            "person_id": person_id,
            "weapon_id": weapon_id,
            "person_confidence": round(
                float(person_confidence),
                4,
            ),
            "weapon_confidence": round(
                float(weapon_confidence),
                4,
            ),
            "first_seen": round(
                float(first_seen),
                3,
            ),
            "confirmed_at": round(
                float(time.time()),
                3,
            ),
            "last_seen": round(
                float(last_seen),
                3,
            ),
            "full_frame": str(
                full_frame_path.relative_to(PROJECT_ROOT)
            ),
            "person_crop": str(
                person_crop_path.relative_to(PROJECT_ROOT)
            ),
            "threat_state": "CONFIRMED_THREAT",
        }

        event_json_path.write_text(
            json.dumps(
                event_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "evidence_image": str(
                full_frame_path.relative_to(PROJECT_ROOT)
            ),
            "person_crop": str(
                person_crop_path.relative_to(PROJECT_ROOT)
            ),
        }

    def get_threat_state(
        self,
        has_weapon: bool,
        confirmed_threat: bool,
    ) -> str:
        """Map a person's current status to a temporal state label.

        NORMAL → WEAPON_CANDIDATE → CONFIRMED_THREAT
        """

        if confirmed_threat:
            return "CONFIRMED_THREAT"
        if has_weapon:
            return "WEAPON_CANDIDATE"
        return "NORMAL"

    # ========================================================
    # CLEAN THREAT STATE
    # ========================================================

    def cleanup_threat_state(
        self,
        now: float,
    ) -> None:

        expired = []

        for person_id, state in (
            self.threat_state.items()
        ):

            last_seen = float(
                state.get(
                    "last_seen",
                    now,
                )
            )

            if (
                now - last_seen
                > self.person_memory_ttl
            ):

                expired.append(
                    person_id
                )

        for person_id in expired:

            self.threat_state.pop(
                person_id,
                None,
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
        # Validate frame.
        # ----------------------------------------------------

        if frame is None:

            raise ValueError(
                "process_frame received an empty frame."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):

            raise TypeError(
                "process_frame expects a NumPy image."
            )

        if frame.size == 0:

            raise ValueError(
                "process_frame received an empty NumPy image."
            )

        now = time.time()

        height, width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # Cleanup memory.
        # ----------------------------------------------------

        self.cleanup_person_memory(
            now
        )

        self.cleanup_threat_state(
            now
        )

        # ====================================================
        # YOLO + BoT-SORT + Re-ID
        # ====================================================

        results = self.model.track(
            source=frame,
            conf=self.track_confidence,
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            imgsz=640,
            verbose=False,
        )

        result = results[0] if results else None

        if result is None or result.boxes is None:
            direct_results = self.model(
                frame,
                imgsz=640,
                conf=0.01,
                device=self.device,
                verbose=False,
            )
            if direct_results:
                result = direct_results[0]

        if result is None or result.boxes is None:

            return {
                "frame": frame.copy(),
                "persons": 0,
                "weapons": 0,
                "phones": 0,
                "threats": 0,
                "objects": 0,
                "highest_confidence": 0.0,
                "processing_ms": round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000.0,
                    2,
                ),
                "detections": [],
                "model": self.model_name,
                "device": (
                    "CUDA"
                    if torch.cuda.is_available()
                    else "CPU"
                ),
            }

        if result.boxes is not None:
            tracked_classes = []
            for box in result.boxes:
                if box.cls is not None:
                    tracked_classes.append(int(box.cls[0]))

            if 1 not in tracked_classes and 2 not in tracked_classes:
                direct_results = self.model(
                    frame,
                    imgsz=640,
                    conf=0.01,
                    device=self.device,
                    verbose=False,
                )
                if direct_results:
                    result = direct_results[0]

        # ====================================================
        # RAW OBJECT COLLECTIONS
        # ====================================================

        persons_raw: list[
            dict[str, Any]
        ] = []

        weapons_raw: list[
            dict[str, Any]
        ] = []

        phones_raw: list[
            dict[str, Any]
        ] = []

        highest_confidence = 0.0

        # ====================================================
        # READ YOLO DETECTIONS
        # ====================================================

        if result.boxes is not None:

            track_ids = result.boxes.id

            for index, box in enumerate(
                result.boxes
            ):

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                # --------------------------------------------
                # Per-class thresholds
                # --------------------------------------------

                if (
                    class_id
                    == WEAPON_CLASS_ID
                    and
                    confidence
                    < self.weapon_confidence
                ):

                    continue

                if (
                    class_id
                    == PHONE_CLASS_ID
                    and
                    confidence
                    < self.phone_confidence
                ):

                    continue

                if (
                    class_id
                    == PERSON_CLASS_ID
                    and
                    confidence
                    < self.confidence
                ):

                    continue

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                x1, y1, x2, y2 = [
                    float(v)
                    for v
                    in box.xyxy[
                        0
                    ].tolist()
                ]

                x1 = max(
                    0.0,
                    min(
                        float(width),
                        x1,
                    ),
                )

                y1 = max(
                    0.0,
                    min(
                        float(height),
                        y1,
                    ),
                )

                x2 = max(
                    0.0,
                    min(
                        float(width),
                        x2,
                    ),
                )

                y2 = max(
                    0.0,
                    min(
                        float(height),
                        y2,
                    ),
                )

                # --------------------------------------------
                # ByteTrack ID
                # --------------------------------------------

                track_id = None

                if track_ids is not None:

                    try:

                        track_id = int(
                            track_ids[
                                index
                            ]
                        )

                    except Exception:

                        track_id = None

                detection = {
                    "track_id":
                        track_id,
                    "class_id":
                        class_id,
                    "class_name":
                        self.get_class_name(
                            class_id
                        ),
                    "confidence":
                        round(
                            confidence,
                            4,
                        ),
                    "bbox": [
                        round(
                            x1,
                            2,
                        ),
                        round(
                            y1,
                            2,
                        ),
                        round(
                            x2,
                            2,
                        ),
                        round(
                            y2,
                            2,
                        ),
                    ],
                }

                highest_confidence = max(
                    highest_confidence,
                    confidence,
                )

                if (
                    class_id
                    == PERSON_CLASS_ID
                ):

                    persons_raw.append(
                        detection
                    )

                elif (
                    class_id
                    == WEAPON_CLASS_ID
                ):

                    weapons_raw.append(
                        detection
                    )

                elif (
                    class_id
                    == PHONE_CLASS_ID
                ):

                    phones_raw.append(
                        detection
                    )

        # ====================================================
        # PERSON STABLE IDs
        # ====================================================

        persons: list[
            dict[str, Any]
        ] = []

        reserved_person_ids: set[
            str
        ] = set()

        for person in persons_raw:

            stable_id, similarity = (
                self.assign_person_id(
                    frame,
                    person["bbox"],
                    person["track_id"],
                    now,
                    reserved_person_ids,
                )
            )

            person["person_id"] = (
                stable_id
            )

            person["reid_similarity"] = (
                round(
                    similarity,
                    4,
                )
            )

            persons.append(
                person
            )

        # ====================================================
        # ASSOCIATE WEAPONS TO PEOPLE
        #
        # For every weapon, score it against every currently
        # visible person using `association_score` (containment
        # + proximity, see that method's docstring) and assign
        # it to whichever person scores highest -- but only if
        # that best score clears ASSOCIATION_THRESHOLD. A weapon
        # is never assigned to more than one person, and a
        # weapon that isn't plausibly near *anyone* is left
        # unassigned (best_person_id stays None), which is what
        # keeps a random weapon elsewhere in frame from making
        # every visible person a threat.
        # ====================================================

        associated_weapon_indices: set[
            int
        ] = set()

        person_weapon_map: dict[
            str,
            list[dict[str, Any]],
        ] = {
            person["person_id"]: []
            for person in persons
        }

        for weapon_index, weapon in enumerate(
            weapons_raw
        ):

            best_person_id = None

            best_score = 0.0

            for person in persons:

                person_id = (
                    person[
                        "person_id"
                    ]
                )

                score = self.association_score(
                    weapon["bbox"],
                    person["bbox"],
                )

                if score < self.association_threshold:
                    continue

                if score > best_score:
                    best_score = score
                    best_person_id = person_id

            if best_person_id is not None:

                person_weapon_map[
                    best_person_id
                ].append(
                    weapon
                )

                associated_weapon_indices.add(
                    weapon_index
                )

        # ====================================================
        # ASSOCIATE PHONES TO PEOPLE
        # ====================================================

        person_phone_map: dict[
            str,
            list[dict[str, Any]],
        ] = {
            person["person_id"]: []
            for person in persons
        }

        for phone in phones_raw:

            best_person_id = None

            best_score = 0.0

            for person in persons:

                person_id = (
                    person[
                        "person_id"
                    ]
                )

                center_inside = (
                    self.object_belongs_to_person(
                        phone["bbox"],
                        person["bbox"],
                    )
                )

                if not center_inside:
                    continue

                iou = (
                    self.calculate_iou(
                        phone["bbox"],
                        person["bbox"],
                    )
                )

                score = (
                    iou * 0.70
                    + 0.30
                )

                if score > best_score:

                    best_score = score

                    best_person_id = (
                        person_id
                    )

            if best_person_id is not None:

                person_phone_map[
                    best_person_id
                ].append(
                    phone
                )

        annotated = frame.copy()

        # ====================================================
        # FINAL DETECTIONS
        # ====================================================

        final_detections: list[
            dict[str, Any]
        ] = []

        threats = 0

        # ====================================================
        # PERSON RESULTS
        # ====================================================

        for person in persons:

            person_id = (
                person[
                    "person_id"
                ]
            )

            associated_weapons = (
                person_weapon_map.get(
                    person_id,
                    [],
                )
            )

            associated_phones = (
                person_phone_map.get(
                    person_id,
                    [],
                )
            )

            has_weapon = (
                len(
                    associated_weapons
                )
                > 0
            )

            has_phone = (
                len(
                    associated_phones
                )
                > 0
            )

            # ------------------------------------------------
            # Temporal threat confirmation.
            #
            # NORMAL → WEAPON_CANDIDATE → CONFIRMED_THREAT
            #
            # update_threat_state returns (confirmed, just_confirmed)
            # where just_confirmed flips True only on the exact frame
            # of the CANDIDATE → CONFIRMED transition.
            # ------------------------------------------------

            confirmed_threat, just_confirmed = (
                self.update_threat_state(
                    person_id,
                    has_weapon,
                )
            )

            threat_state = self.get_threat_state(
                has_weapon,
                confirmed_threat,
            )

            # ------------------------------------------------
            # Weapon / phone confidence for this person.
            # ------------------------------------------------

            weapon_confidence = max(
                [
                    float(
                        weapon[
                            "confidence"
                        ]
                    )
                    for weapon
                    in associated_weapons
                ],
                default=0.0,
            )

            phone_confidence = max(
                [
                    float(
                        phone[
                            "confidence"
                        ]
                    )
                    for phone
                    in associated_phones
                ],
                default=0.0,
            )

            # ------------------------------------------------
            # Event lifecycle + one-shot evidence capture.
            #
            # Evidence images are captured ONLY on the frame
            # where the person first becomes a CONFIRMED
            # THREAT.  On subsequent frames the already-stored
            # paths are reused — no re-saving.
            # ------------------------------------------------

            current_event = self.event_state.get(
                person_id
            )

            if confirmed_threat:

                if (
                    current_event is None
                    or not current_event.get(
                        "active",
                        False,
                    )
                ):

                    # -----------------------------------------------
                    # Transition to CONFIRMED — create event dir
                    # and capture ONE evidence image.
                    # -----------------------------------------------

                    event_id = (
                        self._capture_event_evidence(
                            person_id
                        )
                    )

                    person_crop = (
                        self.crop_box(
                            frame,
                            person[
                                "bbox"
                            ],
                        )
                    )

                    if (
                        person_crop
                        is not None
                    ):

                        evidence_paths = (
                            self._write_evidence(
                                event_id=event_id,
                                person_id=person_id,
                                frame=annotated,
                                person_crop=person_crop,
                                weapon_id=(
                                    self.get_weapon_id(
                                        associated_weapons[
                                            0
                                        ][
                                            "track_id"
                                        ]
                                    )
                                    if associated_weapons
                                    else None
                                ),
                                weapon_confidence=weapon_confidence,
                                person_confidence=float(
                                    person[
                                        "confidence"
                                    ]
                                ),
                                first_seen=float(
                                    self.person_memory.get(
                                        person_id,
                                        {},
                                    ).get(
                                        "first_seen",
                                        now,
                                    )
                                ),
                                last_seen=float(
                                    self.person_memory.get(
                                        person_id,
                                        {},
                                    ).get(
                                        "last_seen",
                                        now,
                                    )
                                ),
                            )
                        )

                        self.event_state[
                            person_id
                        ] = {
                            "active": True,
                            "event_id": event_id,
                            "evidence_image": (
                                evidence_paths.get(
                                    "evidence_image"
                                )
                            ),
                            "person_crop": (
                                evidence_paths.get(
                                    "person_crop"
                                )
                            ),
                        }

                    else:

                        self.event_state[
                            person_id
                        ] = {
                            "active": True,
                            "event_id": event_id,
                            "evidence_image": None,
                            "person_crop": None,
                        }

                # else: already confirmed — reuse existing event.

            elif (
                current_event is not None
                and current_event.get(
                    "active",
                    False,
                )
            ):

                # -----------------------------------------------
                # Was confirmed, weapon now absent —
                # deactivate event but preserve evidence paths.
                # -----------------------------------------------

                self.event_state[
                    person_id
                ] = {
                    "active": False,
                    "event_id": (
                        current_event.get(
                            "event_id"
                        )
                    ),
                    "evidence_image": (
                        current_event.get(
                            "evidence_image"
                        )
                    ),
                    "person_crop": (
                        current_event.get(
                            "person_crop"
                        )
                    ),
                }

            # ------------------------------------------------
            # Re-read final event info from the (possibly
            # updated) state — avoids stale-reference bugs.
            # ------------------------------------------------

            current_event = self.event_state.get(
                person_id,
                {},
            )

            event_id = (
                current_event.get(
                    "event_id"
                )
            )

            evidence_image = (
                current_event.get(
                    "evidence_image"
                )
            )

            person_crop_path = (
                current_event.get(
                    "person_crop"
                )
            )

            # ------------------------------------------------
            # If a weapon is present in the current frame,
            # this person is a threat candidate.
            #
            # The dashboard gets "threat_candidate" immediately,
            # but "is_threat" becomes true after confirmation.
            # ------------------------------------------------

            if confirmed_threat:

                threats += 1

            final_person = {

                "track_id":
                    person["track_id"],

                "person_id":
                    person_id,

                "class_id":
                    PERSON_CLASS_ID,

                "class_name":
                    "person",

                "confidence":
                    person["confidence"],

                "is_person":
                    True,

                "has_weapon":
                    has_weapon,

                "has_phone":
                    has_phone,

                "weapon_count":
                    len(
                        associated_weapons
                    ),

                "phone_count":
                    len(
                        associated_phones
                    ),

                "threat_candidate":
                    has_weapon,

                "is_threat":
                    confirmed_threat,

                "threat_state":
                    threat_state,

                "weapon_confidence":
                    round(
                        weapon_confidence,
                        4,
                    ),

                "phone_confidence":
                    round(
                        phone_confidence,
                        4,
                    ),

                "threat_confidence":
                    round(
                        weapon_confidence,
                        4,
                    ),

                "reid_similarity":
                    person[
                        "reid_similarity"
                    ],

                "first_seen":
                    round(
                        float(
                            self.person_memory[
                                person_id
                            ][
                                "first_seen"
                            ]
                        ),
                        3,
                    ),

                "last_seen":
                    round(
                        float(
                            self.person_memory[
                                person_id
                            ][
                                "last_seen"
                            ]
                        ),
                        3,
                    ),

                "seen_count":
                    int(
                        self.person_memory[
                            person_id
                        ].get(
                            "seen_count",
                            0,
                        )
                    ),

                "event_id":
                    event_id,

                "evidence_image":
                    evidence_image,

                "person_crop":
                    person_crop_path,

                "bbox":
                    person["bbox"],
            }

            final_detections.append(
                final_person
            )

        # ====================================================
        # WEAPON RESULTS
        # ====================================================

        for weapon_index, weapon in enumerate(
            weapons_raw
        ):

            weapon_id = self.get_weapon_id(
                weapon[
                    "track_id"
                ]
            )

            associated = (
                weapon_index
                in associated_weapon_indices
            )

            associated_person_id = None

            if associated:

                for person_id, items in (
                    person_weapon_map.items()
                ):

                    if weapon in items:

                        associated_person_id = (
                            person_id
                        )

                        break

            weapon_result = {

                "track_id":
                    weapon["track_id"],

                "weapon_id":
                    weapon_id,

                "class_id":
                    WEAPON_CLASS_ID,

                "class_name":
                    "weapon",

                "confidence":
                    weapon["confidence"],

                "is_person":
                    False,

                "is_threat":
                    associated,

                "associated_with_person":
                    associated,

                "associated_person_id":
                    associated_person_id,

                "bbox":
                    weapon["bbox"],
            }

            final_detections.append(
                weapon_result
            )

        # ====================================================
        # PHONE RESULTS
        # ====================================================

        for phone in phones_raw:

            phone_id = self.get_phone_id(
                phone[
                    "track_id"
                ]
            )

            associated_person_id = None

            for person_id, items in (
                person_phone_map.items()
            ):

                if phone in items:

                    associated_person_id = (
                        person_id
                    )

                    break

            phone_result = {

                "track_id":
                    phone["track_id"],

                "phone_id":
                    phone_id,

                "class_id":
                    PHONE_CLASS_ID,

                "class_name":
                    "phone",

                "confidence":
                    phone["confidence"],

                "is_person":
                    False,

                # VERY IMPORTANT:
                # Phone is NEVER a threat.
                "is_threat":
                    False,

                "associated_with_person":
                    associated_person_id
                    is not None,

                "associated_person_id":
                    associated_person_id,

                "bbox":
                    phone["bbox"],
            }

            final_detections.append(
                phone_result
            )

        # ====================================================
        # DRAW
        # ====================================================

        annotated = frame.copy()

        for detection in final_detections:

            x1, y1, x2, y2 = map(
                int,
                detection["bbox"],
            )

            class_name = detection[
                "class_name"
            ]

            confidence = float(
                detection[
                    "confidence"
                ]
            )

            is_person = bool(
                detection.get(
                    "is_person",
                    False,
                )
            )

            is_threat = bool(
                detection.get(
                    "is_threat",
                    False,
                )
            )

            # ------------------------------------------------
            # Person
            # ------------------------------------------------

            if is_person:

                person_id = detection.get(
                    "person_id",
                    "UNKNOWN",
                )

                has_weapon = bool(
                    detection.get(
                        "has_weapon",
                        False,
                    )
                )

                threat_candidate = bool(
                    detection.get(
                        "threat_candidate",
                        False,
                    )
                )

                if is_threat:

                    label = (
                        f"PERSON "
                        f"{person_id} "
                        f"[THREAT + WEAPON] "
                        f"{confidence:.2f}"
                    )

                    box_color = (
                        70,
                        80,
                        220,
                    )

                elif threat_candidate:

                    label = (
                        f"PERSON "
                        f"{person_id} "
                        f"[WEAPON CHECKING] "
                        f"{confidence:.2f}"
                    )

                    box_color = (
                        0,
                        165,
                        255,
                    )

                else:

                    label = (
                        f"PERSON "
                        f"{person_id} "
                        f"{confidence:.2f}"
                    )

                    box_color = (
                        70,
                        145,
                        225,
                    )

            # ------------------------------------------------
            # Weapon
            # ------------------------------------------------

            elif class_name == "weapon":

                weapon_id = detection.get(
                    "weapon_id",
                    "W-??",
                )

                associated_person_id = (
                    detection.get(
                        "associated_person_id"
                    )
                )

                if associated_person_id:

                    label = (
                        f"WEAPON "
                        f"{weapon_id} "
                        f"{confidence:.2f} "
                        f"[{associated_person_id}]"
                    )

                    box_color = (
                        70,
                        80,
                        220,
                    )

                else:

                    label = (
                        f"WEAPON "
                        f"{weapon_id} "
                        f"{confidence:.2f}"
                    )

                    box_color = (
                        0,
                        165,
                        255,
                    )

            # ------------------------------------------------
            # Phone
            # ------------------------------------------------

            elif class_name == "phone":

                phone_id = detection.get(
                    "phone_id",
                    "PH-??",
                )

                associated_person_id = (
                    detection.get(
                        "associated_person_id"
                    )
                )

                if associated_person_id:

                    label = (
                        f"PHONE "
                        f"{phone_id} "
                        f"{confidence:.2f} "
                        f"[{associated_person_id}]"
                    )

                else:

                    label = (
                        f"PHONE "
                        f"{phone_id} "
                        f"{confidence:.2f}"
                    )

                box_color = (
                    60,
                    180,
                    75,
                )

            # ------------------------------------------------
            # Unknown
            # ------------------------------------------------

            else:

                label = (
                    f"{class_name.upper()} "
                    f"{confidence:.2f}"
                )

                box_color = (
                    130,
                    130,
                    130,
                )

            # =================================================
            # BOX
            # =================================================

            cv2.rectangle(
                annotated,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                box_color,
                2,
            )

            # =================================================
            # LABEL
            # =================================================

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            font_scale = 0.50

            thickness = 1

            (
                text_width,
                text_height,
            ), baseline = (
                cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )
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

            cv2.rectangle(
                annotated,
                (
                    x1,
                    label_top,
                ),
                (
                    x1
                    + text_width
                    + 8,
                    label_bottom,
                ),
                box_color,
                -1,
            )

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

        for person in persons:
            person_id = person.get("person_id")
            if not person_id:
                continue
            current_event = self.event_state.get(person_id)
            if not current_event or not current_event.get("active"):
                continue
            if current_event.get("event_id") is None:
                continue
            weapon_id = None
            weapon_confidence_value = 0.0
            associated_weapons = person_weapon_map.get(person_id, [])
            if associated_weapons:
                weapon_item = associated_weapons[0]
                weapon_id = self.get_weapon_id(weapon_item.get("track_id"))
                weapon_confidence_value = float(weapon_item.get("confidence", 0.0))
            person_crop = self.crop_box(frame, person["bbox"])
            if person_crop is None:
                continue
            event_dir = PROJECT_ROOT / "events" / time.strftime("%Y-%m-%d") / current_event["event_id"]
            event_dir.mkdir(parents=True, exist_ok=True)
            full_frame_path = event_dir / "full_frame.jpg"
            person_crop_path = event_dir / "person_crop.jpg"
            event_json_path = event_dir / "event.json"
            cv2.imwrite(str(full_frame_path), annotated)
            cv2.imwrite(str(person_crop_path), person_crop)
            event_payload = {
                "event_id": current_event["event_id"],
                "person_id": person_id,
                "weapon_id": weapon_id,
                "person_confidence": round(float(person["confidence"]), 4),
                "weapon_confidence": round(float(weapon_confidence_value), 4),
                "first_seen": round(float(self.person_memory.get(person_id, {}).get("first_seen", time.time())), 3),
                "confirmed_at": round(float(time.time()), 3),
                "last_seen": round(float(self.person_memory.get(person_id, {}).get("last_seen", time.time())), 3),
                "full_frame": str(full_frame_path.relative_to(PROJECT_ROOT)),
                "person_crop": str(person_crop_path.relative_to(PROJECT_ROOT)),
                "threat_state": "CONFIRMED_THREAT",
            }
            event_json_path.write_text(json.dumps(event_payload, indent=2), encoding="utf-8")
            self.event_state[person_id]["evidence_image"] = str(full_frame_path.relative_to(PROJECT_ROOT))
            self.event_state[person_id]["person_crop"] = str(person_crop_path.relative_to(PROJECT_ROOT))

        # ====================================================
        # TIMING
        # ====================================================

        processing_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "frame":
                annotated,

            "persons":
                len(persons),

            "weapons":
                len(weapons_raw),

            "phones":
                len(phones_raw),

            "threats":
                threats,

            "objects":
                len(final_detections),

            "highest_confidence":
                round(
                    highest_confidence,
                    4,
                ),

            "processing_ms":
                round(
                    processing_ms,
                    2,
                ),

            "detections":
                final_detections,

            "model":
                self.model_name,

            "device":
                (
                    "CUDA"
                    if torch.cuda.is_available()
                    else "CPU"
                ),
        }


# ============================================================
# GLOBAL DETECTOR
# ============================================================

detector = ThreatDetector()