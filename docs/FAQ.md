# Frequently Asked Questions

## What does the model detect?

It draws a bounding box around a person and assigns one of two contextual classes: `normal_person` or `potentially_dangerous_person`.

## Does a detection prove that someone is dangerous?

No. A detection reflects visual similarity to the training labels. It does not prove identity, intent, guilt, or future behavior.

## Why was YOLO11s selected?

YOLO11s achieved the best aggregate precision, F1, mAP50, and mAP50-95 in the reported held-out test comparison, while remaining smaller than YOLO11m and YOLO11l.

## Where is the dataset?

Use the Google Drive link in the README or download the versioned dataset archive from the v1.0.0 GitHub release.

## Where are the trained weights?

The v1.0.0 release includes best and last checkpoints for YOLO11n, YOLO11s, the YOLO11s preprocessing variant, YOLO11m, and YOLO11l.

## Can I reproduce every published number exactly?

Not necessarily. The original notebooks, locked environment, random seed, and resolved optimizer details were not present in the supplied workspace. The repository provides a transparent reconstruction and documents this limitation.

## Can this be deployed commercially?

The open-source project and Ultralytics integration use AGPL-3.0. Closed-source or commercial deployment may require an Ultralytics enterprise license, as well as separate legal, privacy, safety, and dataset-rights review.

## How should results be reported?

State the dataset split, software and hardware versions, confidence threshold, precision, recall, F1, mAP50, mAP50-95, class-level results, limitations, and representative failures.
