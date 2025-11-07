from ultralytics import YOLO
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

checkpoint = "runs_food_uec100_a10g/train2/weights/last.pt"

model = YOLO("yolov8n.pt")
#model = YOLO(checkpoint) # to resume training from last saved model
model.train(
    data="data/processed/uecfood100/uec_food_labels.yaml",
    resume=True, patience=10,
    imgsz=640, epochs=30, device=0,
    batch=64, workers=4, cache=False, amp=True,
    cos_lr=True, compile=False, close_mosaic=10,
    project="runs_food_uec100_a10g",
)
