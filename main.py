from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
# Optionally, adjust speech rate or volume:
# engine.setProperty("rate", 150)
# engine.setProperty("volume", 0.9)

# Load the YOLO model
model = YOLO("yolo11n.pt")  # Update this path to your model file if needed

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Variables to control speech announcements
last_message = ""
last_announcement_time = time.time()
announcement_interval = 3  # seconds between announcements

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions for region determination
    frame_height, frame_width = frame.shape[:2]

    # Run object detection on the frame
    results = model(frame)

    # Annotate the frame showing detection boxes (for debugging purposes)
    annotated_frame = results[0].plot()

    # Dictionaries to record detected objects per region
    regions = {"left": set(), "center": set(), "right": set()}
    center_danger = False  # flag for an immediate threat ahead

    # Process detection results (if boxes are found)
    if results[0].boxes is not None:
        boxes = results[0].boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            # Skip weak detections
            if conf < 0.5:
                continue

            # Calculate the horizontal center of the bounding box
            center_x = (x1 + x2) / 2

            # Divide the frame into three horizontal parts:
            # left, center, and right.
            if center_x < frame_width / 3:
                region = "left"
            elif center_x < 2 * frame_width / 3:
                region = "center"
            else:
                region = "right"

            label = model.names[int(class_id)]
            regions[region].add(label)

            # For the center region, check if the object is very wide,
            # which might indicate it is close (hence more dangerous).
            if region == "center":
                box_width_ratio = (x2 - x1) / frame_width
                if box_width_ratio > 0.3:
                    center_danger = True

    # Build a verbal message based on the detections and their regions
    message_parts = []
    if center_danger:
        message_parts.append("WARNING: Immediate obstacle ahead.")
    else:
        if regions["center"]:
            message_parts.append("Obstacle ahead: " + ", ".join(regions["center"]))
        if regions["left"]:
            message_parts.append("On your left: " + ", ".join(regions["left"]))
        if regions["right"]:
            message_parts.append("On your right: " + ", ".join(regions["right"]))

    # If nothing is detected, notify that the path is clear.
    if not any(regions.values()) and not center_danger:
        message_parts.append("Path is clear.")

    # Join the message parts into one string.
    message = " | ".join(message_parts)

    current_time = time.time()
    # Announce if the message is different from the previous one or
    # a minimal time interval has passed.
    if message != last_message or (
        current_time - last_announcement_time > announcement_interval
    ):
        print(message)  # For logging or debugging purposes
        engine.say(message)
        engine.runAndWait()
        last_message = message
        last_announcement_time = current_time

    # Display the annotated frame (for debugging).
    # In a production system (or on wearable glasses) you might disable this.
    cv2.imshow("Obstacle Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
