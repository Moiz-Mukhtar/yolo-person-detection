import cv2
from ultralytics import YOLO

def run_person_detection(input_video_path: str, output_video_path: str, conf_threshold: float = 0.5):
    # 1. Load the pretrained YOLO11 Nano model (Slide 12)
    model = YOLO("yolo11n.pt")

    # 2. Open input video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video_path}")

    # Read video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Fallback FPS if metadata is unreadable

    # 3. Setup VideoWriter to save processed video (Slide 12)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing: {input_video_path} -> {output_video_path} at {fps:.2f} FPS")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run detection: classes=[0] filters for 'person' (Slides 9, 10, 12)
        # Drop boxes below the confidence threshold (Slide 7)
        results = model(frame, classes=[0], conf=conf_threshold, verbose=False)

        # 5. Extract boxes and draw manually with OpenCV (Slide 12)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box corners: x1, y1, x2, y2 (Slide 7)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Label format: "person <confidence>" (Slide 11)
                label = f"person {confidence:.2f}"

                # Draw bounding box rectangle (Slide 12)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

                # Draw label background & text (Slide 12)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1, label_size[1] + 10)
                cv2.rectangle(
                    frame,
                    (x1, label_y - label_size[1] - 8),
                    (x1 + label_size[0] + 4, label_y + 4),
                    (0, 140, 255),
                    cv2.FILLED
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        # Write frame to destination file
        out.write(frame)

    # Clean up resources
    cap.release()
    out.release()
    print("Video processing complete.")

if __name__ == "__main__":
    # Ensure you place a sample video named 'input.mp4' in the project directory
    run_person_detection(input_video_path="input.mp4", output_video_path="output.mp4")