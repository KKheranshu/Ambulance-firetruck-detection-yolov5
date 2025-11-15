import cv2
import torch

# ✅ Load your trained model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)
model.conf = 0.6  # confidence threshold

# 🚀 Start webcam
cap = cv2.VideoCapture(0)

AMBULANCE_LABEL = "ambulance"  # Make sure this matches your trained label

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    mid_x = w // 2

    # ✂️ Split frame into left and right
    left_lane = frame[:, :mid_x]
    right_lane = frame[:, mid_x:]

    # 🔍 Detect objects in both lanes
    results_left = model(left_lane)
    results_right = model(right_lane)

    det_left = results_left.pandas().xyxy[0]
    det_right = results_right.pandas().xyxy[0]

    # 🔢 Count vehicles / detect ambulance
    left_vehicles = len(det_left)
    right_vehicles = len(det_right)

    left_ambulance = any(det_left["name"] == AMBULANCE_LABEL)
    right_ambulance = any(det_right["name"] == AMBULANCE_LABEL)

    # 🚦 Decision logic
    if left_ambulance:
        left_status = "GREEN → Ambulance!"
        right_status = "RED"
        left_color, right_color = (0, 255, 0), (0, 0, 255)
    elif right_ambulance:
        right_status = "GREEN → Ambulance!"
        left_status = "RED"
        right_color, left_color = (0, 255, 0), (0, 0, 255)
    elif left_vehicles > right_vehicles:
        left_status = f"GREEN → {left_vehicles} Vehicles"
        right_status = "RED"
        left_color, right_color = (0, 255, 0), (0, 0, 255)
    elif right_vehicles > left_vehicles:
        right_status = f"GREEN → {right_vehicles} Vehicles"
        left_status = "RED"
        right_color, left_color = (0, 255, 0), (0, 0, 255)
    else:
        left_status = right_status = "YELLOW → Equal Traffic"
        left_color = right_color = (0, 255, 255)

    # 🖥️ Combine frames back for display
    combined = cv2.hconcat([left_lane, right_lane])

    # 🏷️ Overlay status
    cv2.putText(combined, left_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, left_color, 3)
    cv2.putText(combined, right_status, (mid_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, right_color, 3)

    cv2.imshow("2-Lane Traffic System", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
