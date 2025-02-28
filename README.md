# Obstacle Detection Assistive Device for the Visually Impaired

This project is designed to assist visually impaired individuals by providing real-time audio feedback for obstacle detection using a YOLO-based object recognition system. The application uses a webcam to capture live video, performs object detection using a YOLO model (via the Ultralytics library), and divides the frame spatially into left, center, and right regions. Based on the detected objects and their positions, the system issues voice announcements to alert the user—especially emphasizing warnings when an obstacle is directly ahead.

## Features

- **Real-Time Object Detection**  
  Utilizes a YOLO model from the Ultralytics package to detect objects from a live webcam feed.

- **Spatial Awareness**  
  Splits the camera frame into three sections (left, center, and right) to provide location-based feedback on obstacles.

- **Immediate Danger Alerts**  
  Provides an urgent verbal warning if large objects are detected in the center of the frame that may indicate an immediate hazard.

- **Voice-Based Feedback**  
  Uses the `pyttsx3` text-to-speech engine to announce obstacles, helping users understand their surroundings without needing to see a display.

- **Configurable Parameters**  
  Adjust settings such as detection confidence thresholds, speech intervals, and spatial crisis thresholds to balance responsiveness and user comfort.

## Prerequisites

- **Python 3.7+**

Install the required libraries using pip:

```bash
pip install ultralytics opencv-python pyttsx3
