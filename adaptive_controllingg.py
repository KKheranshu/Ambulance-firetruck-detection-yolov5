import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device


# Email alert function with GPS map snapshot (fixed location for Thapar University)
def send_alert_to_hospital(location, image_path):
    sender_email = "smart.traffic.alertss@gmail.com"
    sender_password = "kzzdwqqiilyacnyi"  # Replace with your app password
    recipient_email = "khuwahish298@gmail.com"  # Replace with your actual test email

    subject = "🚨 Emergency Alert: Ambulance Detected"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fixed_location = "Thapar University, Patiala"
    body = f"Ambulance detected near: {fixed_location}.\nTime: {timestamp}\nMap: https://www.google.com/maps/search/?api=1&query=Thapar+University+Patiala"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent to hospital.")
    except Exception as e:
        print("❌ Email sending failed:", e)


# Device setup
device = select_device("mps" if torch.backends.mps.is_available() else "cpu")
model = DetectMultiBackend("runs/train/emergency-detector-v5/weights/best.pt", device=device)
model.conf = 0.40
print("Model loaded. Classes:", model.names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found.")
    exit()

frame_id = 0


def get_lane_regions(W):
    return [(0, int(W / 2)), (int(W / 2), W)]


def is_in_lane(cx, region):
    return region[0] <= cx < region[1]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue

    H, W = frame.shape[:2]
    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.40, iou_thres=0.4)[0]

    labels = model.names
    lane_counts = [0, 0]
    lane_regions = get_lane_regions(W)
    ambulance_detected = False

    if pred is not None and len(pred):
        for det in pred:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            class_name = labels[int(cls)].lower()
            cx = (x1 + x2) / 2

            for i, region in enumerate(lane_regions):
                if is_in_lane(cx, region):
                    if class_name in ["car", "bus", "truck"]:
                        lane_counts[i] += 1
                    elif class_name == "ambulance":
                        if not ambulance_detected:
                            ambulance_detected = True
                            snapshot = f"amb_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(snapshot, frame)
                            send_alert_to_hospital("Thapar University, Patiala", snapshot)

                    color = (0, 255, 255) if i == 0 else (0, 165, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if ambulance_detected:
        decision = "Ambulance Detected — Give Green to All"
    elif lane_counts[0] > lane_counts[1]:
        decision = "Lane 1: GREEN | Lane 2: RED"
    elif lane_counts[1] > lane_counts[0]:
        decision = "Lane 1: RED | Lane 2: GREEN"
    else:
        decision = "Both Lanes Balanced"

    cv2.rectangle(frame, (lane_regions[0][0], 0), (lane_regions[0][1], H), (255, 0, 0), 2)
    cv2.rectangle(frame, (lane_regions[1][0], 0), (lane_regions[1][1], H), (0, 0, 255), 2)
    cv2.putText(
        frame, f"Lane 1 Vehicles: {lane_counts[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"Lane 2 Vehicles: {lane_counts[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(frame, decision, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    cv2.imshow("Smart Traffic System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
