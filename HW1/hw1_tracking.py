from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "./argoverse.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, show=True, classes=[2], save=True)

        # Get the boxes, track IDs, class
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()


        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        annotator = Annotator(frame, line_width=2, example=str(model.names))

        # Plot the tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Rewrite the bounding box and the text on it
            label = model.model.names[int(cls)] + f" {track_id}"
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            annotator.box_label([x1, y1, x2, y2], label=str(label), color=(255, 0, 255))
            annotated_frame = annotator.im

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=3)
            # Draw the middle point
            cv2.circle(annotated_frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (255, 0, 255), -1)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()