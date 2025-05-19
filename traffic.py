import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Setup paths
image_folder = "images"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Get all image files
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
print("Found files:", image_files)

vehicle_classes = {2: "car", 3: "2-wheeler", 5: "bus", 7: "truck", 1: "2-wheeler"}

# DataFrame for summary
df = pd.DataFrame(columns=["way", "Image", "Total Vehicles", "Cars", "2-Wheelers", "Buses", "Trucks"])

colors = {
    "car": (255, 0, 0),
    "2-wheeler": (0, 255, 0),
    "bus": (0, 0, 255),
    "truck": (255, 255, 0)
}

# Run inference and annotate
for idx, image_path in enumerate(image_files):
    img = cv2.imread(image_path)
    results = model(img)[0]

    vehicle_count = {"car": 0, "2-wheeler": 0, "bus": 0, "truck": 0}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in vehicle_classes:
            label = vehicle_classes[cls_id]
            vehicle_count[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)

    total = sum(vehicle_count.values())
    df = pd.concat([df, pd.DataFrame.from_records([{
        "way": f"Way {idx + 1}",
        "Image": os.path.basename(image_path),
        "Total Vehicles": total,
        "Cars": vehicle_count["car"],
        "2-Wheelers": vehicle_count["2-wheeler"],
        "Buses": vehicle_count["bus"],
        "Trucks": vehicle_count["truck"]
    }])], ignore_index=True)

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

# Assign dummy red light times
red_light_times = [30, 40, 35, 25][:len(df)]
df["Red Light Time (s)"] = red_light_times
df["Traffic Flow (vehicles/sec)"] = df["Total Vehicles"] / df["Red Light Time (s)"]

# Calculate weight score
def vehicle_weight_score(row):
    return (row["Buses"] * 4 + row["Trucks"] * 3 + row["Cars"] * 2 + row["2-Wheelers"])

df["Weight Score"] = df.apply(vehicle_weight_score, axis=1)

# Determine priority way
priority_way = df.sort_values(by=["Total Vehicles", "Weight Score", "Traffic Flow (vehicles/sec)"], ascending=False).iloc[0]

print("\n📊 Traffic Summary Table:")
print(df[["way", "Total Vehicles", "Red Light Time (s)", "Traffic Flow (vehicles/sec)", "Weight Score"]])

# Random Forest Prediction
X = df[["Traffic Flow (vehicles/sec)", "Red Light Time (s)", "Weight Score", "Total Vehicles"]]
y = df["Red Light Time (s)"] + df["Traffic Flow (vehicles/sec)"] * 10  # Simulated label

model_rf = RandomForestRegressor()
model_rf.fit(X, y)

# Predict for priority way
x_priority = pd.DataFrame([priority_way[["Traffic Flow (vehicles/sec)", "Red Light Time (s)", "Weight Score", "Total Vehicles"]]])
predicted_green_time = model_rf.predict(x_priority)[0]

print(f"\n🚦 Priority Lane: {priority_way['way']}")
print(f"✅ Predicted Green Light Duration: {predicted_green_time:.2f} seconds")

# Save results
df.to_csv("vehicle_detection_results.csv", index=False)
print("\n✅ Results saved to 'vehicle_detection_results.csv'")
