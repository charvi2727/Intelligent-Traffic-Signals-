# Install necessary packages (uncomment if needed)
# pip install ultralytics opencv-python numpy pandas matplotlib

import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can switch to yolov8s.pt, yolov8m.pt for more accuracy

# Setup paths
image_folder = "images"  # Folder with input images
output_folder = "output"  # Folder to save output images with boxes
os.makedirs(output_folder, exist_ok=True)

# Get all image files
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print("Found files:", image_files)

# COCO vehicle class IDs
vehicle_classes = {2: "car", 3: "2-wheeler", 5: "bus", 7: "truck", 1: "2-wheeler"}

# DataFrame for summary
df = pd.DataFrame(columns=["Image", "Total Vehicles", "Cars", "2-Wheelers", "Buses", "Trucks"])

# Define colors for drawing
colors = {
    "car": (255, 0, 0),
    "2-wheeler": (0, 255, 0),
    "bus": (0, 0, 255),
    "truck": (255, 255, 0)
}

# Run inference and annotate
for image_path in image_files:
    img = cv2.imread(image_path)
    results = model(img)[0]

    vehicle_count = {"car": 0, "2-wheeler": 0, "bus": 0, "truck": 0}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in vehicle_classes:
            label = vehicle_classes[cls_id]
            vehicle_count[label] += 1

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)

    total = sum(vehicle_count.values())
    df = pd.concat([df, pd.DataFrame.from_records([{
        "Image": os.path.basename(image_path),
        "Total Vehicles": total,
        "Cars": vehicle_count["car"],
        "2-Wheelers": vehicle_count["2-wheeler"],
        "Buses": vehicle_count["bus"],
        "Trucks": vehicle_count["truck"]
    }])], ignore_index=True)

    # Save annotated image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

# Save results
df.to_csv("vehicle_detection_results.csv", index=False)
print("Results saved to 'vehicle_detection_results.csv'.")
print("Annotated images saved in the 'output/' folder.")

# Show top results
print(df.head())
