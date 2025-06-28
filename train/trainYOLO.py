from ultralytics import YOLO
model = YOLO('yolov8x.pt')
results = model(image_path)
detections = results[0].boxes.xyxy  # if you want to count or filter
