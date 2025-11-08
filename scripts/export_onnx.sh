# 1) Export ONNX from trained PyTorch weights
yolo export model=/home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.pt \
  format=onnx imgsz=640 opset=12 dynamic=True simplify=True

# 2) Quick Sanity with ONNX Runtime
pip install onnxruntime-gpu
yolo predict model=/home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.onnx \
  source=/home/ubuntu/calorify/data/processed/uecfood100/images/val \
  imgsz=640 conf=0.33 device=0
