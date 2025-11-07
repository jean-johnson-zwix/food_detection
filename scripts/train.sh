pip install -U ultralytics onnx onnxsim onnxruntime-gpu

# verify GPU
python - <<'PY'
import torch; print("CUDA OK:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0))
PY

# initial training
yolo detect train \
  model=yolov8s.pt \
  data=data/processed/uecfood100/uec_food_labels.yaml \
  imgsz=768 epochs=150 batch=auto workers=4 cache=True amp=True cos_lr=True \
  compile=True close_mosaic=10 device=0 project=runs_food_uec100_a10g
