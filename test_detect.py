import torch

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Run inference on an image
results = model('path_to_your_image.jpg')
results.show()

