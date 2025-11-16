import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import yaml
import gradio as gr
from ultralytics import YOLO

ENGINE = "/home/ubuntu/calorify/saved_models/best_tensorrt_fp16.engine"   # TensorRT FP16 engine
DATA_YAML = "/home/ubuntu/calorify/data/processed/uecfood100/uec_food_labels.yaml"
IMGSZ     = 640
DEF_CONF  = 0.33
DEF_IOU   = 0.60
SERVER    = dict(server_name="0.0.0.0", server_port=8000)        # will serve at :8000
TITLE     = "Food Detection using YOLOv8 TensorRT (FP16)"
DESC      = "Live webcam food detection (UEC Food-100). Engine: TensorRT FP16."

def load_names(yaml_path: str, model=None) -> List[str] | None:
    if yaml_path and Path(yaml_path).exists():
        y = yaml.safe_load(open(yaml_path, "r"))
        names = y.get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        return names
    return getattr(model, "names", None)

print(f"[load] TensorRT engine: {ENGINE}")
MODEL = YOLO(ENGINE)

try:
    _ = MODEL.predict(source=np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8),
                      imgsz=IMGSZ, conf=DEF_CONF, iou=DEF_IOU, verbose=False)
except Exception as e:
    raise SystemExit(f"[fatal] Failed to load TensorRT engine: {e}")

NAMES = load_names(DATA_YAML, MODEL) or [f"class_{i}" for i in range(100)]

def detect(frame: np.ndarray, conf: float, iou: float):
    if frame is None:
        return None, pd.DataFrame(), "No frame"

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    t0 = time.perf_counter()
    res = MODEL.predict(source=frame, imgsz=IMGSZ, conf=conf, iou=iou, verbose=False)[0]
    total_ms = (time.perf_counter() - t0) * 1000.0

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return frame, pd.DataFrame(columns=["class","confidence","x1","y1","x2","y2"]), f"No detections • {total_ms:.2f} ms"

    # Convert BGR to RGB
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    xyxy  = boxes.xyxy.cpu().numpy()
    clses = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    rows = []
    for (x1, y1, x2, y2), c, cf in zip(xyxy, clses, confs):
        cname = NAMES[int(c)] if int(c) < len(NAMES) else f"class_{int(c)}"
        rows.append({
            "class": cname,
            "confidence": float(cf),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
        })
    df = pd.DataFrame(rows)

    return annotated_rgb, df, f"Detections: {len(rows)} • {total_ms:.2f} ms"

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"## {TITLE}\n{DESC}\n\n**Engine:** `{ENGINE}`  \n**Classes:** {len(NAMES)}")
    with gr.Row():
        with gr.Column(scale=2):
            cam = gr.Image(sources=["webcam"], type="numpy", label="Webcam")
            with gr.Row():
                conf_in = gr.Slider(0.05, 0.95, value=DEF_CONF, step=0.01, label="Confidence")
                iou_in  = gr.Slider(0.10, 0.95, value=DEF_IOU, step=0.01, label="IoU")
        with gr.Column(scale=3):
            out_img = gr.Image(type="numpy", label="Annotated")
            out_df  = gr.Dataframe(headers=["class","confidence","x1","y1","x2","y2"], label="Detections", interactive=False)
            out_txt = gr.Markdown()

    if hasattr(cam, "stream"):
        cam.stream(detect, inputs=[cam, conf_in, iou_in], outputs=[out_img, out_df, out_txt])
    else:
        cam.change(detect, inputs=[cam, conf_in, iou_in], outputs=[out_img, out_df, out_txt])

if __name__ == "__main__":
    demo.launch(**SERVER, share=False, inbrowser=False)
