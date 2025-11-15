🚦 Smart Traffic System – Ambulance & Firetruck Detection (YOLOv5)

This project is a Smart Traffic Management System that detects Ambulances and Firetrucks using YOLOv5.
When an emergency vehicle is detected, the system:

Automatically turns the corresponding lane’s traffic signal GREEN

Sends a simple text alert and an email notification to the nearby hospital

Helps reduce emergency response time during critical situations

This is part of an AI + IoT project designed for real-time intelligent traffic control.

⭐ Features

🚑 Real-time detection of ambulances & firetrucks

🚦 Automatic traffic signal control for emergency lane

📩 Email alert system for nearby hospitals

🎥 Works with webcam, iPhone camera, or CCTV feed

⚙️ Trained using a custom YOLOv5 dataset

🧊 Easy to integrate with Arduino, Raspberry Pi, or any IoT setup

📊 Script for traffic density analysis (optional)

🧠 Tech Stack

YOLOv5 (PyTorch)

Python

OpenCV

SMTP / smtplib for email alerts

Arduino / Raspberry Pi (optional)

Numpy & Pandas

📝 System Workflow

A camera captures the live traffic feed

YOLOv5 model identifies ambulances and firetrucks

If an emergency vehicle is detected:

Corresponding traffic signal is instantly turned GREEN

Normal signals are paused

A text alert + email is sent to the hospital

Once the vehicle passes, signals return to normal timing