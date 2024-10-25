import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utilis import YOLO_Detection, label_detection, draw_working_areas

def setup_device():
    """Check if CUDA is available and set the device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device
    
def load_yolo_model(device):
    """Load the YOLO model and configure it."""
    model = YOLO("yolov8n.pt")
    model.to(device)
    model.nms = 0.5
    print(f"Model classes: {model.names}")
    return model

source_video = "/home/yahya/Downloads/work-desk.mp4"
cap = cv2.VideoCapture(source_video)


# Define working areas (To draw polygons coords)
working_area = [
    [(499, 41), (384, 74), (377, 136), (414, 193), (417, 112), (548, 91)],  # Area 0
    [(547, 91), (419, 113), (414, 189), (452, 289), (453, 223), (615, 164)],  # Area 1
    [(158, 84), (294, 85), (299, 157), (151, 137)],  # Area 2
    [(151, 139), (300, 155), (321, 251), (143, 225)],  # Area 3
    [(143, 225), (327, 248), (351, 398), (142, 363)],  # Area 4
    [(618, 166), (457, 225), (454, 289), (522, 396), (557, 331), (698, 262)]   # Area 5
]

# Initialize variables
time_in_area = {index: 0 for index in range(len(working_area))}  # Time tracker for each area
frame_duration = 0.1  # Duration of each frame in seconds
entry_time = {}  # Track entry time for each detected object
frame_cnt = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_cnt += 1

    # Get YOLO detections
    boxes, classes, names, confidences, ids = YOLO_Detection(model, frame, conf=0.05, mode="track")
    polygon_detections = [False] * len(working_area)  # Track detections in each polygon

    for box, cls, id in zip(boxes, classes, ids):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_point = (int(center_x), int(center_y))

        # Label the detection
        label_detection(frame=frame, text=f"{names[int(cls)]}, {int(id)}", tbox_color=(255, 144, 30), left=x1, top=y1, bottom=x2, right=y2)

        # Check if the detection point is inside any working polygon
        for index, pos in enumerate(working_area):
            matching_result = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
            if matching_result >= 0:
                polygon_detections[index] = True  # Mark this polygon as having a detection

                # Track entry time for the detected worker
                if id not in entry_time:
                    # First detection of this ID
                    entry_time[id] = (frame_cnt, index)  # Store the frame count and area index
                else:
                    start_frame, area_index = entry_time[id]
                    if area_index != index:  # Object has entered a different area
                        # Add time spent in the previous area
                        time_in_area[area_index] += frame_duration
                        print(f"Object ID {id} left Area {area_index + 1}. Time counted: {time_in_area[area_index]:.2f}s")
                        # Update to the new area
                        entry_time[id] = (frame_cnt, index)  # Update entry time to new area
                    else:
                        # Still in the same area
                        time_in_area[area_index] += frame_duration  # Increment time in seconds
                        # Debug statement to verify time counting in Area 6
                        if area_index == 5:  # Area 6 corresponds to index 5
                            print(f"Object ID {id} is in Area 6. Time counted: {time_in_area[area_index]:.2f}s")

    # Draw polygons based on detection status
    for index, pos in enumerate(working_area):
        if not polygon_detections[index]:  # No detection in this polygon
            draw_working_areas(frame=frame, area=pos, index=index, color=(0, 0, 255))  # Draw in red
        else:
            draw_working_areas(frame=frame, area=pos, index=index)  # Draw in green

    # Draw the transparent box for time spent in each area
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 250), (255, 255, 255), -1)  # White background
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)  # Make it transparent

    # Display time spent in each area
    for index in range(len(working_area)):
        time_spent = time_in_area[index]
        cv2.putText(frame, f"Cabin {index + 1}: {round(time_spent)}s", (15, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    # Show the frame with overlays
    cv2.imshow('Frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup code if necessary
cap.release()
cv2.destroyAllWindows()
