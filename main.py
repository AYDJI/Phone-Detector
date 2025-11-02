import argparse
import time
import cv2
from ultralytics import YOLO
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Detect phones from camera using YOLO (ultralytics).")
    p.add_argument("--cam", type=int, default=0, help="Camera index (default=0).")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0..1).")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model to use (default yolov8n.pt).")
    p.add_argument("--no-show", action="store_true", help="Disable preview window.")
    args = p.parse_args()
    args.show = not args.no_show  # Default is True unless you add --no-show
    return args


def draw_box(frame, xyxy, label, score):
    x1, y1, x2, y2 = map(int, xyxy)
    # Draw rectangle and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (12, 150, 255), 2)
    text = f"{label} {score:.2f}"
    # calculate text size
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 6, y1), (12, 150, 255), -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def main():
    args = parse_args()
    # Load model (will download yolov8n.pt if not present)
    print("Loading model:", args.model)
    model = YOLO(args.model)  # ultralytics YOLO object

    # Open camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {args.cam}")
        return

    print("Camera opened. Press 'q' to quit.")
    # We'll track fps
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            frame_count += 1
            # Run inference on the single frame. Set verbose=False to reduce console spam.
            # The model returns a list of Results, but passing a single frame returns [results]; we grab [0].
            results = model(frame, conf=args.conf, verbose=False)[0]

            # results.boxes is a Boxes object; iterate through detections
            detected_phone = False
            for box in results.boxes:
                # box.cls (tensor), box.conf, box.xyxy
                cls_idx = int(box.cls.cpu().numpy().astype(int))  # class index
                score = float(box.conf.cpu().numpy())
                name = model.names.get(cls_idx, str(cls_idx))

                # The common COCO class name for phones is "cell phone" (or similar).
                # We'll accept any class name that contains the word 'phone' (lowercased) to be flexible.
                if "phone" in name.lower() or "cell phone" in name.lower():
                    xyxy = box.xyxy.cpu().numpy().flatten()  # [x1,y1,x2,y2]
                    draw_box(frame, xyxy, name, score)
                    detected_phone = True

            # Show an indicator text
            status_text = "PHONE DETECTED" if detected_phone else "no phone"
            cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0) if detected_phone else (0,0,255), 2, cv2.LINE_AA)

            # compute fps occasionally
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10 / (now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            if args.show:
                cv2.imshow("Phone Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # If not showing, still print status to console
                print(f"[{frame_count}] {status_text}")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exiting.")

if __name__ == "__main__":
    main()
