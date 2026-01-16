from ultralytics import YOLO

# Choose a pretrained starting point (recommended) or a model config
# Examples: "yolov8n.pt", "yolov8s.pt", etc.
model = YOLO("yolo11n.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",          # use 0 for first GPU, or "cpu"
    workers=8,
    project="runs",
    name="exp_yolo"
)

metrics = model.val(
    data="data.yaml",
    split="val",       # uses valid/images
    imgsz=640,
    device=0
)

print(metrics)
