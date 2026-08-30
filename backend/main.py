from __future__ import annotations

import asyncio
import base64
import binascii
import json

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.detector import detector


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Threat Detection API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ConfidenceRequest(BaseModel):

    value: float = Field(
        ge=0.01,
        le=0.99,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name":
            "Threat Detection API",

        "status":
            "running",

        "model":
            detector.model_name,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status":
            "ok",

        "model_name":
            detector.model_name,

        "classes":
            detector.names,

        "class_count":
            detector.class_count,

        "confidence":
            detector.confidence,

        "device":
            detector.device,

        "device_name":
            detector.device_name,
    }


# ============================================================
# MODEL
# ============================================================

@app.get("/api/model")
def model_info():

    return {
        "model_name":
            detector.model_name,

        "classes":
            detector.names,

        "class_count":
            detector.class_count,

        "device":
            detector.device,

        "device_name":
            detector.device_name,
    }


# ============================================================
# CONFIDENCE
# ============================================================

@app.post("/api/confidence")
def update_confidence(
    request: ConfidenceRequest,
):

    detector.set_confidence(
        request.value
    )

    return {
        "confidence":
            detector.confidence,
    }


# ============================================================
# FRAME DECODER
# ============================================================

def decode_frame(
    frame_data: str,
) -> np.ndarray | None:

    # Remove data URL prefix.
    if "," in frame_data:

        frame_data = (
            frame_data
            .split(",", 1)[1]
        )

    try:

        image_bytes = (
            base64.b64decode(
                frame_data,
                validate=True,
            )
        )

    except (
        ValueError,
        binascii.Error,
    ):

        return None

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    return frame


# ============================================================
# WEBSOCKET DETECTION
# ============================================================

@app.websocket("/ws/detect")
async def detection_socket(
    websocket: WebSocket,
):

    await websocket.accept()

    client = websocket.client

    print(
        f"[WS] Client connected: {client}"
    )

    try:

        while True:

            # ------------------------------------------------
            # RECEIVE
            # ------------------------------------------------

            raw_message = (
                await websocket.receive_text()
            )

            try:

                payload = json.loads(
                    raw_message
                )

            except json.JSONDecodeError:

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "Invalid JSON payload.",
                        }
                    )
                )

                continue


            # ------------------------------------------------
            # FRAME
            # ------------------------------------------------

            frame_data = payload.get(
                "frame"
            )

            if not isinstance(
                frame_data,
                str,
            ):

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "Frame field missing.",
                        }
                    )
                )

                continue


            # ------------------------------------------------
            # DECODE
            # ------------------------------------------------

            frame = decode_frame(
                frame_data
            )

            if frame is None:

                print(
                    "[WS] Frame decode failed."
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "Could not decode image.",
                        }
                    )
                )

                continue


            print(
                "[WS] Frame received: "
                f"{frame.shape[1]}x"
                f"{frame.shape[0]}"
            )


            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            try:

                result = await asyncio.to_thread(
                    detector.process_frame,
                    frame,
                )

            except Exception as exc:

                print(
                    f"[YOLO ERROR] {exc}"
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "YOLO inference failed.",
                        }
                    )
                )

                continue


            print(
                "[YOLO] "
                f"{result['processing_ms']} ms | "
                f"persons={result['persons']} | "
                f"threats={result['threats']}"
            )


            # ------------------------------------------------
            # FRAME
            # ------------------------------------------------

            annotated_frame = result.pop(
                "frame"
            )


            success, encoded_image = (
                cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        55,
                    ],
                )
            )


            if not success:

                await websocket.send_text(
                    json.dumps(
                        {
                            "type":
                                "error",

                            "message":
                                "Image encoding failed.",
                        }
                    )
                )

                continue


            encoded_frame = (
                base64.b64encode(
                    encoded_image.tobytes()
                ).decode(
                    "ascii"
                )
            )


            # ------------------------------------------------
            # RESPONSE
            # ------------------------------------------------

            response = {

                "type":
                    "frame",

                "frame":
                    (
                        "data:image/jpeg;base64,"
                        + encoded_frame
                    ),

                "persons":
                    result.get(
                        "persons",
                        0,
                    ),

                "threats":
                    result.get(
                        "threats",
                        0,
                    ),

                "objects":
                    result.get(
                        "objects",
                        0,
                    ),

                "highest_confidence":
                    result.get(
                        "highest_confidence",
                        0,
                    ),

                "processing_ms":
                    result.get(
                        "processing_ms",
                        0,
                    ),

                "detections":
                    result.get(
                        "detections",
                        [],
                    ),

                "model":
                    result.get(
                        "model",
                        detector.model_name,
                    ),

                "device":
                    result.get(
                        "device",
                        "CPU",
                    ),
            }


            await websocket.send_text(
                json.dumps(
                    response
                )
            )


    except Exception as exc:

        print(
            f"[WS] Connection ended: {exc}"
        )


    finally:

        print(
            f"[WS] Client disconnected: {client}"
        )