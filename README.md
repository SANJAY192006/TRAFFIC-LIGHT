# Smart Traffic Light System

This is a Python project for an AI-powered smart traffic light timing system using computer vision and machine learning.

## Features
- Real-time vehicle detection using YOLOv8
- Machine learning prediction for green light duration
- Emergency vehicle override
- Console-based traffic light simulation
- Optional Tkinter GUI for live signal colors
- Matplotlib bar graph for vehicle counts

## Requirements
- Python 3.8+
- Libraries: opencv-python, numpy, scikit-learn, matplotlib, ultralytics, torch

## Installation
1. Clone or download the project.
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Place a video file named `traffic.mp4` in the project directory, or use webcam (default if video not found).
2. Run the script: `python traffic_light_system.py`
3. The system will start detecting vehicles, predicting timings, and simulating lights.

## Notes
- YOLOv8 model will be downloaded automatically on first run.
- Emergency vehicles are assumed to be detected as 'bus' or 'truck' for demo purposes.
- The system updates every 5 seconds.
