import os
import cv2
import time
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker  # Ensure you have a tracker.py with a Tracker class
import easyocr  # For OCR
import tkinter as tk
from tkinter import filedialog

# ==================== Configuration ====================

# Paths to models
VEHICLE_MODEL_PATH = 'yolov8s.pt'  # Path to YOLO vehicle detection model
HELMET_MODEL_PATH = "/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles/best.pt"  # Path to helmet detection model

# CSV for violations
CSV_FILENAME = 'violations.csv'
CSV_COLUMNS = ['Timestamp', 'License Plate', 'Violation', 'Fine (INR)']

# Detection parameters
SPEED_LIMIT_KMH = 50  # Speed limit in km/h
DISTANCE_BETWEEN_LINES_METERS = 10  # Distance between the two lines in meters

# Line positions (y-coordinates)
RED_LINE_Y = 198
BLUE_LINE_Y = 268
OFFSET = 6  # Tolerance in pixels

# ==================== File Selection ====================

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

# Prompt the user to select the input video file
INPUT_VIDEO_PATH = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", ".mp4"), ("AVI files", ".avi")])

if not INPUT_VIDEO_PATH:
    raise IOError("No video file selected")

OUTPUT_VIDEO_PATH = 'output.avi'

# ==================== Initialize Components ====================

# Initialize YOLO models
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
if not os.path.isfile(HELMET_MODEL_PATH):
    raise FileNotFoundError(f"Helmet model file not found: {HELMET_MODEL_PATH}")
helmet_model = YOLO(HELMET_MODEL_PATH)

# Initialize OCR
reader = easyocr.Reader(['en'])

# Initialize Tracker
tracker = Tracker()

# Class list (ensure it matches your YOLO model's classes)
CLASS_LIST = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    # ... other classes ...
]

# Initialize video capture
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_VIDEO_PATH}")

# Get the original video frame width and height
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

# Initialize CSV
if not os.path.exists(CSV_FILENAME):
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_FILENAME, index=False)

# Initialize variables for speed detection
down = {}
counter_down = []
up = {}
counter_up = []

# Initialize frame count
count = 0

# Create folders to save detected frames and license plates
DETECTED_FRAMES_DIR = 'detected_frames'
LICENSE_PLATES_DIR = 'license_plates'
os.makedirs(DETECTED_FRAMES_DIR, exist_ok=True)
os.makedirs(LICENSE_PLATES_DIR, exist_ok=True)

# ==================== Helper Functions ====================

def log_violation(timestamp, license_plate, violation, fine):
    """Logs the violation to the CSV file."""
    df = pd.DataFrame([{
        'Timestamp': timestamp,
        'License Plate': license_plate,
        'Violation': violation,
        'Fine (INR)': fine
    }])
    df.to_csv(CSV_FILENAME, mode='a', header=False, index=False)

def detect_license_plate(frame, x1, y1, x2, y2, vehicle_id):
    """Detects and returns the license plate text using OCR and saves the license plate image."""
    roi = frame[y1:y2, x1:x2]
    license_plate = reader.readtext(roi)
    if license_plate:
        plate_text = license_plate[0][-2]  # Assuming the second last element is the text
        # Save the license plate image
        plate_image_path = os.path.join(LICENSE_PLATES_DIR, f'vehicle_{vehicle_id}_plate.jpg')
        cv2.imwrite(plate_image_path, roi)
        return plate_text
    return "Unknown"

def detect_helmet(frame, x1, y1, x2, y2):
    """Detects if a helmet is present in the given region."""
    roi = frame[y1:y2, x1:x2]
    results = helmet_model.predict(roi)
    if results and len(results) > 0:
        for result in results:
            if 'helmet' in result.names:
                return True
    return False

# ==================== Main Processing Loop ====================

frame_retry_count = 0  # Counter to track retry attempts for frame reads

while True:
    ret, frame = cap.read()

    # Retry if frame not successfully read (this handles potential frame skipping)
    if not ret:
        frame_retry_count += 1
        if frame_retry_count >= 10:  # Allow up to 10 retries for missing frames
            break
        else:
            continue  # Try to read the next frame again
    frame_retry_count = 0  # Reset retry counter if a frame is successfully read

    count += 1

    # Run YOLO detection for vehicles
    results = vehicle_model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []

    # Prepare bounding boxes for tracker
    bboxes = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det[:6]
        class_id = int(class_id)
        class_name = CLASS_LIST[class_id] if class_id < len(CLASS_LIST) else 'Unknown'
        if class_name in ['car', 'motorcycle']:
            w = x2 - x1
            h = y2 - y1
            bboxes.append([int(x1), int(y1), int(w), int(h)])

    # Update tracker
    tracked_vehicles = tracker.update(bboxes)

    # Process each tracked vehicle
    for vehicle in tracked_vehicles:
        x1, y1, w, h, vehicle_id = vehicle
        cx = int((x1 + x1 + w) // 2)
        cy = int((y1 + y1 + h) // 2)

        # Speed Detection
        speed = 0
        current_time = time.time()

        if RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            down[vehicle_id] = current_time

        if vehicle_id in down and BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            elapsed_time = current_time - down[vehicle_id]
            if elapsed_time > 0:
                speed = (DISTANCE_BETWEEN_LINES_METERS / elapsed_time) * 3.6
                if speed > SPEED_LIMIT_KMH and vehicle_id not in counter_down:
                    license_plate = detect_license_plate(frame, x1, y1, x1 + w, y1 + h, vehicle_id)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                    log_violation(timestamp, license_plate, 'Overspeeding', 1000)
                    counter_down.append(vehicle_id)

        if class_name == 'motorcycle':
            helmet_present = detect_helmet(frame, x1, y1, x1 + w, y1 + h)
            if not helmet_present and vehicle_id not in counter_down:
                license_plate = detect_license_plate(frame, x1, y1, x1 + w, y1 + h, vehicle_id)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                log_violation(timestamp, license_plate, 'No Helmet', 500)
                counter_down.append(vehicle_id)

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {vehicle_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, 'Helmet Detected' if helmet_present else 'No Helmet',
                        (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Show the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' or 'Esc' is pressed
    key = cv2.waitKey(1)  # Reduced wait time to maintain high frame rate
    if key == ord('q') or key == 27:  # 27 is the Esc key
        break

# ==================== Cleanup ====================
cap.release()
out.release()
cv2.destroyAllWindows()