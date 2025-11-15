import cv2
import torch
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression

# Email alert function with attachment
def send_alert_to_hospital(location, image_path):
    sender_email = "smart.traffic.alertss@gmail.com"
    sender_password = "kzzdwqqiilyacnyi"  # Replace with actual app password
    recipient_email = "kashyapkheranshu@gmail.com"  # Replace with your test email

    subject = "🚨 Emergency Alert: Ambulance Detected"
    body = f"Ambulance detected at {location}. Please prepare emergency team and equipment and doctors ready for the treatment.\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(image_path)}",
        )
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email with image sent to hospital.")
    except Exception as e:
        print("❌ Failed to send email:", e)

# Select device
device = select_device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load model
model_path = 'runs/train/emergency-detector-v5/weights/best.pt'
model = DetectMultiBackend(model_path, device=device)
model.conf = 0.60  # Increased confidence threshold to reduce false positives
print("Loaded model classes:", model.names)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue

    H, W = frame.shape[:2]
    lane_region = (0, 0, W, H)  # full screen single lane

    img = letterbox(frame, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.60, iou_thres=0.4)[0]

    labels = model.names
    ambulance_detected = False

    if pred is not None and len(pred):
        for det in pred:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            class_name = labels[int(cls)].lower()

            print(f"📌 DETECTED: {class_name} | Confidence: {conf:.2f}")

            if class_name != "ambulance":
                continue

            ambulance_detected = True
            snapshot_path = f"ambulance_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_path, frame)
            send_alert_to_hospital("Sector 21 Red Light", snapshot_path)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    decision = "Default: GREEN"
    if ambulance_detected:
        decision = "🚨 Ambulance Detected — GIVE GREEN"

    # Draw lane and decision
    cv2.rectangle(frame, lane_region[:2], lane_region[2:], (255, 0, 0), 2)
    cv2.putText(frame, "Lane 1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, decision, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Smart Traffic System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
