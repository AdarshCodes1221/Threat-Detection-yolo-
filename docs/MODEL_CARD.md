# Model Card

## Model Details

| Field | Value |
| --- | --- |
| Selected model | YOLO11s |
| Task | Two-class object detection |
| Input size | 832 pixels |
| Training batch | 8 |
| Epochs set, completed | 150, 132 |
| Early-stopping patience | 35 |
| Recorded device | NVIDIA Tesla T4 |
| Demonstration confidence | 0.35 |

The model localizes a person and assigns `normal_person` or `potentially_dangerous_person`. It is a graduation-thesis research artifact built with Ultralytics YOLO.

## Test-Set Results

| Variant | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.81 | 0.69 | 0.74 | 0.78 | 0.60 |
| YOLO11s | **0.87** | 0.69 | **0.77** | **0.80** | **0.64** |
| YOLO11s preprocessing variant | 0.78 | 0.71 | 0.74 | 0.79 | 0.62 |
| YOLO11m | 0.78 | 0.69 | 0.73 | 0.76 | 0.59 |
| YOLO11l | 0.80 | **0.72** | 0.76 | 0.79 | 0.63 |

At confidence 0.20, YOLO11s recorded class-level F1 values of 0.73 for the normal-person class and 0.81 for the potentially-dangerous-person class. Its class-level mAP50-95 values were 0.49 and 0.80, respectively.

## Selection Rationale

YOLO11s led precision, aggregate F1, mAP50, and mAP50-95 while remaining smaller than YOLO11m and YOLO11l. YOLO11l achieved the highest recall, but its lower precision and similar mAP did not justify selecting the larger model for the thesis demonstration.

## Intended Use

The model is intended for controlled academic analysis and human-reviewed demonstrations. A detection should only identify a frame for further review.

## Limitations

- Recall of 0.69 means relevant instances may be missed.
- False positives may occur with small, occluded, poorly lit, or weapon-like objects.
- Performance may shift across cameras, locations, populations, weather, crowd density, compression, and operational thresholds.
- The current evaluation does not establish real-time latency, calibration, robustness, fairness, privacy compliance, or operational safety.

## License

The model integration and released trained artifacts are provided under AGPL-3.0. Commercial or closed-source deployment may require a separate Ultralytics enterprise license. Dataset terms apply separately.
