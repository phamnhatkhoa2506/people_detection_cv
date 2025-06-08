from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolov8n')
    results = model.train(data='People-Detection-1/data.yaml', epochs=10)