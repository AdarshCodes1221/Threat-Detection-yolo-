# Reproducibility Guide

## Scope

This guide reconstructs the settings recorded in the final thesis. Exact numerical reproduction is not guaranteed because the supplied workspace did not include the original training notebooks, a locked Python environment, the random seed, the exact Ultralytics build, or the detailed choice made by optimizer `auto`.

## Environment

- Python 3.10 or later is recommended.
- The recorded training device was an NVIDIA Tesla T4.
- Install `requirements.txt` in an isolated environment.
- Record the resolved package versions and GPU driver before a new run.

## Dataset Preparation

1. Download `Data-DATN-22207062.zip` from the v1.0.0 release or use the linked Drive folder.
2. Verify the release checksum.
3. Extract the archive.
4. Copy the inner `Dataset` directory to `data/dataset`.
5. Confirm that `train`, `valid`, and `test` each contain matching `images` and `labels` directories.
6. Do not change the split before attempting replication.

## Training

Run one recorded variant at a time:

```bash
python scripts/train.py --variant yolo11n --data config/data.yaml
python scripts/train.py --variant yolo11s --data config/data.yaml
python scripts/train.py --variant yolo11s_preprocessing --data config/data.yaml
python scripts/train.py --variant yolo11m --data config/data.yaml
python scripts/train.py --variant yolo11l --data config/data.yaml
```

The script applies 150 configured epochs, batch size 8, image size 832, optimizer `auto`, and the recorded patience for each model. The preprocessing variant also applies the parameters listed in `config/experiments.yaml`.

## Evaluation

Use the held-out test split only after training and model selection decisions are complete. Record:

- precision and recall;
- aggregate and class-level F1;
- mAP50 and mAP50-95;
- confusion matrix;
- confidence threshold;
- hardware and software versions; and
- failures involving occlusion, scale, lighting, crowd density, and hard negatives.

## Inference

The thesis demonstration used confidence 0.35 for qualitative images. Test-set class-level F1 values were also reported at confidence 0.20. Do not compare results without recording the threshold.

## Reproducibility Checklist

- [ ] Dataset archive and checksum recorded.
- [ ] Split counts match 2,512 train, 314 validation, and 314 test images.
- [ ] Class indices match the published mapping.
- [ ] Package, Python, CUDA, driver, and GPU versions recorded.
- [ ] Seed and deterministic settings recorded for the new run.
- [ ] Configuration and command saved with results.
- [ ] Test data remains isolated from training and tuning.
- [ ] Quantitative and qualitative failures documented.
