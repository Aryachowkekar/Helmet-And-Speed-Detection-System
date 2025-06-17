from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import time
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import easyocr

app = Flask(__name__)

# Ensure you have a 'uploads' directory for video files
UPLOAD_FOLDER = '/home/arya/Desktop/Vs Code/websec/Final/Speed-detection-of-vehicles'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        process_video(filepath)  # Call your video processing function here
        return redirect(url_for('index'))

def process_video(video_path):
    # Your existing video processing code goes here
    pass

if __name__ == '__main__':
    app.run(debug=True)
