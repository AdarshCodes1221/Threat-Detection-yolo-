# Dataset Card

## Summary

The dataset supports a two-class object-detection study in public-area imagery. Bounding boxes surround people rather than weapons alone.

| Field | Value |
| --- | --- |
| Images | 3,140 |
| Instances | 10,595 |
| Train, validation, test | 2,512, 314, 314 images |
| Class 0 | `normal_person` |
| Class 1 | `potentially_dangerous_person` |
| Annotation format | YOLO normalized bounding boxes |
| Recorded export | Roboflow, April 22, 2026 |
| Recorded license | CC BY 4.0 |

## Class Definition

`potentially_dangerous_person` is a contextual annotation for a person who appears with an object such as a firearm or knife in the source frame. It is not an identity, personality, demographic, medical, or criminal label. `normal_person` represents other localized people under the dataset rules.

## Composition

| Split | Images | Normal-person instances | Potentially-dangerous-person instances | Total instances |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,512 | 6,170 | 2,298 | 8,468 |
| Validation | 314 | 968 | 356 | 1,324 |
| Test | 314 | 539 | 264 | 803 |
| Total | 3,140 | 7,677 | 2,918 | 10,595 |

The thesis reports 247 self-captured images from public areas and hard-negative examples involving visually confusable objects such as books, snack packages, bottles, phones, and umbrellas.

## Access

- [Google Drive folder](https://drive.google.com/drive/folders/1m-BzFvQwwWzvTCXApwJftQU3tRt0CmCC?usp=drive_link)
- [GitHub v1.0.0 release](https://github.com/Nambekai/dangerous-person-detection-yolo11/releases/tag/v1.0.0)
- [Recorded Roboflow source](https://universe.roboflow.com/062transinam-s-workspace/do-an-ahpc0)

## Appropriate Uses

- Academic comparison of object-detection models.
- Reproduction studies with documented assumptions.
- Controlled research on false positives, false negatives, occlusion, lighting, and hard negatives.

## Inappropriate Uses

- Biometric identification or face recognition.
- Automated conclusions about intent, guilt, or threat status.
- Predictive policing, population scoring, or demographic inference.
- Deployment without lawful authority, privacy review, human oversight, and local validation.

## Known Limitations

- The class distribution is imbalanced: 72.46 percent normal-person instances and 27.54 percent potentially-dangerous-person instances.
- Public-source imagery may contain geographic, camera, lighting, context, and demographic sampling biases.
- A single frame cannot reliably establish intent or distinguish every harmless object from a weapon.
- The dataset card does not replace a source-by-source rights, consent, privacy, and retention review.

## Attribution

The archive states CC BY 4.0 and identifies the Roboflow project `Do an`. Cite this repository, the recorded Roboflow source, and any underlying source that applies to reused images.
