import cv2
import torch
import os
from collections import defaultdict

# Load your trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/emergency-detector-v5/weights/best.pt',
                       force_reload=True)
model.conf = 0.85  # Higher confidence to reduce false detections

# Start webcam
cap = cv2.VideoCapture(0)

LANE_COUNT = 2
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    lane_width = width // LANE_COUNT

    results = model(frame)
    detections = results.xyxy[0]
    labels = model.names

    lane_counts = [0] * LANE_COUNT
    ambulance_detected = [False] * LANE_COUNT
    total_vehicle_count = 0
    type_count = defaultdict(int)

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cx = (x1 + x2) // 2
        class_id = int(cls)
        class_name = labels[class_id]

        lane_idx = min(cx // lane_width, LANE_COUNT - 1)

        if class_name in ['ambulance', 'firetruck']:
            ambulance_detected[lane_idx] = True
        elif class_name in vehicle_classes:
            lane_counts[lane_idx] += 1
            total_vehicle_count += 1
            type_count[class_name] += 1

            # ✅ Print and speak vehicle type
            print(f"Detected a {class_name}")
            os.system(f"say Detected a {class_name}")

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw 2 lanes and lane counts
    green_lane = None
    for i in range(LANE_COUNT):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width
        color = (0, 0, 255)  # Red

        if ambulance_detected[i]:
            green_lane = i
            color = (0, 255, 0)

        cv2.rectangle(frame, (x_start, 0), (x_end, height), color, 2)
        cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (x_start + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Decision logic
    if green_lane is not None:
        decision = f"🚨 GREEN to Lane {green_lane+1} (Emergency)"
        red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
    else:
        max_count = max(lane_counts)
        if max_count > 0:
            green_lane = lane_counts.index(max_count)
            decision = f"🚗 GREEN to Lane {green_lane+1} (Most Traffic)"
            red_lanes = [str(i+1) for i in range(LANE_COUNT) if i != green_lane]
        else:
            decision = "🟡 No traffic detected"
            red_lanes = []

    # Display GREEN/RED logic
    y_text = height - 90
    if green_lane is not None:
        cv2.putText(frame, f"🟢 GREEN: Lane {green_lane+1}", (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_text += 30
        if red_lanes:
            cv2.putText(frame, f"🔴 RED: Lane(s) {', '.join(red_lanes)}", (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, decision, (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show total vehicle count
    cv2.putText(frame, f"Total Vehicles: {total_vehicle_count}", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Type-wise count
    y_offset = height - 100
    for vtype, count in type_count.items():
        cv2.putText(frame, f"{vtype.capitalize()}: {count}", (width - 220, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset -= 25

    # Show live frame
    cv2.imshow("Smart Traffic (2 Lanes)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
