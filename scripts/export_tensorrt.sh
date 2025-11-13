# Approach: 1 - Start an NVIDIA TensorRT container and build the TensorRT engine -> model dedicated to this container
#cd /home/ubuntu/calorify
#docker run --gpus all -it --rm -v $PWD:/ws -w /ws nvcr.io/nvidia/tensorrt:24.10-py3

# Inside the container, build engine from ONNX
# trtexec --onnx=/ws/saved_models/best.onnx  --saveEngine=/ws/saved_models/best_fp16.engine  --fp16 --memPoolSize=workspace:8192  --shapes=images:1x3x640x640  --minShapes=images:1x3x640x640  --optShapes=images:1x3x640x640  --maxShapes=images:4x3x640x640  --verbose
# trtexec --loadEngine=/ws/saved_models/best_fp16.engine  --shapes=images:1x3x640x640  --warmUp=200 --duration=15 --streams=1

# Approach 2 - Export to TensorRT engine (integrate with Ultralytics workflow)
yolo export model=/home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.pt \
  format=engine imgsz=640 half=True device=0

mv /home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.engine \
   /home/ubuntu/calorify/saved_models/best_tensorrt_fp16.engine