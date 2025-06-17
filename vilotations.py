import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import easyocr  # For OCR

# Initialize models
model = YOLO('yolov8s.pt')  # Vehicle detection model
helmet_model_path = '/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles/best.pt'  # Update this path to the location of your helmet detection model

# Check if the helmet model file exists
if not os.path.isfile(helmet_model_path):
    raise FileNotFoundError(f"Model file not found: {helmet_model_path}")

helmet_model = YOLO(helmet_model_path)  # Load the pre-trained helmet detection model
reader = easyocr.Reader(['en'])  # Initialize OCR

# Class list and tracker initialization
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']
tracker = Tracker()
cap = cv2.VideoCapture('/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles/highway.mp4')  # Update this path to the location of your video file

# Violation counters and storage
violations = []
down = {}

# Define lines for checking speed
red_line_y = 198
blue_line_y = 268
speed_limit = 50  # Speed limit in km/h

# Prepare CSV file for recording violations
csv_filename = 'violations.csv'
csv_columns = ['License Plate', 'Violation', 'Fine (INR)']
if not os.path.exists(csv_filename):
    pd.DataFrame(columns=csv_columns).to_csv(csv_filename, index=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLO detection for vehicles
    results = model.predict(frame)
    detections = results[0].boxes.data.detach().cpu().numpy()

    # Track vehicles
    bboxes = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = map(int, det[:6])
        class_name = class_list[class_id]
        if class_name == 'car':
            bboxes.append([x1, y1, x2, y2])

    tracked_vehicles = tracker.update(bboxes)

    for vehicle in tracked_vehicles:
        x1, y1, x2, y2, vehicle_id = vehicle
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Check speed and record if over the limit
        if red_line_y < cy < red_line_y + 10:
            down[vehicle_id] = time.time()
        if vehicle_id in down and blue_line_y < cy < blue_line_y + 10:
            elapsed = time.time() - down[vehicle_id]
            speed = (10 / elapsed) * 3.6
            if speed > speed_limit:
                license_plate = reader.readtext(frame[y1:y2, x1:x2])
                license_text = license_plate[0][-2] if license_plate else "Unknown"
                violations.append({'License Plate': license_text, 'Violation': 'Speeding', 'Fine (INR)': 1000})

        # Detect helmets and record if absent
        helmet_results = helmet_model.predict(frame[y1:y2, x1:x2])
        for helmet_result in helmet_results:
            if 'no_helmet' in helmet_result:
                license_plate = reader.readtext(frame[y1:y2, x1:x2])
                license_text = license_plate[0][-2] if license_plate else "Unknown"
                violations.append({'License Plate': license_text, 'Violation': 'No Helmet', 'Fine (INR)': 500})

        # You can add more violation checks here (e.g., wrong U-turn)

    # Display the frame with annotations
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save violations to CSV
if violations:
    pd.DataFrame(violations).to_csv(csv_filename, mode='a', header=False, index=False)

cap.release()
cv2.destroyAllWindows()
