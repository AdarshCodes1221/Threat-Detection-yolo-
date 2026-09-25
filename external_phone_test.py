from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL = r".\models\person_weapon_phone_best.pt"
IMG_DIR = Path(r"D:\Downloads\cellphone_test\test\images")
LBL_DIR = Path(r"D:\Downloads\cellphone_test\test\labels")

model = YOLO(MODEL)

total_gt = 0
total_pred = 0
matched = 0
ious = []

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    a1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    a2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = a1 + a2 - inter

    return inter / union if union > 0 else 0


for img_path in sorted(IMG_DIR.iterdir()):

    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    label_path = LBL_DIR / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))

    if img is None or not label_path.exists():
        continue

    h, w = img.shape[:2]

    gt_boxes = []

    for line in label_path.read_text().splitlines():
        p = line.strip().split()

        if len(p) != 5:
            continue

        # External dataset has ONLY phone = class 0
        xc, yc, bw, bh = map(float, p[1:])

        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h

        gt_boxes.append([x1,y1,x2,y2])

    total_gt += len(gt_boxes)

    result = model.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.10,
        verbose=False
    )[0]

    pred_boxes = []

    for box, cls, conf in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy()
    ):
        # YOUR MODEL: class 2 = phone
        if int(cls) == 2:
            pred_boxes.append(box.tolist())

    total_pred += len(pred_boxes)

    used = set()

    for gt in gt_boxes:

        best_iou = 0
        best_idx = -1

        for i, pred in enumerate(pred_boxes):

            if i in used:
                continue

            score = iou(gt, pred)

            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou >= 0.50:
            matched += 1
            used.add(best_idx)
            ious.append(best_iou)

print()
print("============================================")
print("EXTERNAL PHONE TEST")
print("============================================")
print("Images tested       :", len(list(IMG_DIR.glob("*"))))
print("Ground-truth phones :", total_gt)
print("Predicted phones    :", total_pred)
print("Matched @ IoU 0.50  :", matched)

if total_gt:
    recall = matched / total_gt
    print("Phone Recall        :", round(recall, 4))

if total_pred:
    precision = matched / total_pred
    print("Phone Precision     :", round(precision, 4))

if ious:
    print("Mean matched IoU    :", round(sum(ious)/len(ious), 4))

print("============================================")
