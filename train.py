from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Dataset/weapon_dataset/data.yaml",
    epochs=50,
    device=0,
    imgsz=640
)