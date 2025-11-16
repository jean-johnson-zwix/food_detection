# Food Detection with YOLO

## Demo Video



## Implementation

### Step 1: Data Loading & Processing

#### Dataset: UECFood100

Description: The "UEC FOOD 100" contains 100-kind food photos. Each food photo has a bounding box indicating the location of the food item in the photo.
Most of the food categories in this dataset are popular foods in Japan.

Platform: Kaggle
Link: https://www.kaggle.com/datasets/rkuo2000/uecfood100

#### Commands

```
source /opt/pytorch/bin/activate
python3 -m src.load_data
python3 -m src.process_data
```

### Step 2: Train the YOLO model

### Commands

Train using the Shell Script

```
chmod 700 ./scripts/train.sh
./scripts/train.sh
```

Train using the Python Script
```
python3 -m src.train_yolo
```

### Step 3: Export ONNX model

ONNX is an open format built to represent machine learning models. It allows trained models to be exported and used across frameworks.

```
chmod 700 scripts/export_onnx.sh 
./scripts/export_onnx.sh
```

### Step 4: Start the NVIDIA TensorRT container and convert ONNX to TensorRT engine

NVIDIA TensorRT is an ecosystem of tools for developers to achieve high-performance deep learning inference. They deliver low latency and high throughput.

#### Approach 1: Build the engine with NVIDIA TensorRT container

```
docker run --gpus all -it --rm -v /home/ubuntu:/home/ubuntu -w /home/ubuntu nvcr.io/nvidia/tensorrt:24.10-py3

# commands for inside the container

trtexec --onnx=/ws/saved_models/best.onnx  --saveEngine=/ws/saved_models/best_fp16.engine  --fp16 --memPoolSize=workspace:8192  --shapes=images:1x3x640x640  --minShapes=images:1x3x640x640  --optShapes=images:1x3x640x640  --maxShapes=images:4x3x640x640  --verbose

trtexec --loadEngine=/ws/saved_models/best_fp16.engine  --shapes=images:1x3x640x640  --warmUp=200 --duration=15 --streams=1
```

#### Approach 2: Export the engine  with YOLO

```
# Export to TensorRT engine (integrate with Ultralytics workflow)
yolo export model=/home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.pt \
  format=engine imgsz=640 half=True device=0

mv /home/ubuntu/calorify/runs_food_uec100_a10g/train2/weights/best.engine \
   /home/ubuntu/calorify/saved_models/best_tensorrt_fp16.engine
```

### Step 5: Evaluate the Engine

```
chmod 700 evaluate_engine.sh 
./evaluate_engine.sh
```

### Step 6: Result Evaluation

#### ONNX vs FP16

```
python3 -m src.food_detection
```

Run the food_detection module with both ONNX and FP16 engines.

#####  Results

# Food Detection (ONNX vs FP16) — Run Summary

| run_name            | engine_type     | engine_file               |   files |   imgsz |   conf |   iou |   avg_ms |   throughput_ips |   total_time_s |
|:--------------------|:----------------|:--------------------------|--------:|--------:|-------:|------:|---------:|-----------------:|---------------:|
| ONNX Runtime run    | ONNX Runtime    | best.onnx                 |    1514 |     640 |   0.33 |   0.6 |   72.973 |            13.7  |        110.48  |
| TensorRT (FP16) run | TensorRT (FP16) | best_tensorrt_fp16.engine |    1514 |     640 |   0.33 |   0.6 |    7.845 |           127.47 |         11.878 |


[!Latency Bar Chart](results\onnx_vs_fp16\comparitive_analysis_results\latency_ms_bar.png)

[!Throughput Bar Chart](results\onnx_vs_fp16\comparitive_analysis_results\throughput_ips_bar.png)

