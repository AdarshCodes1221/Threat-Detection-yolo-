import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open laptop camera.")

print("Webcam started.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.35,
        device=0,
        verbose=False,
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Threat Detection - Local Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()