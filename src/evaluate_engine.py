
#!/usr/bin/env python3
"""
eval_engine_noargs.py - end-to-end inference metrics using Ultralytics on a TensorRT .engine
All values are hardcoded below. No CLI args required.
"""

import json, os, time
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

ENGINE   = "/home/ubuntu/calorify/saved_models/best_fp16.engine"
SOURCE   = "/home/ubuntu/calorify/data/processed/uecfood100/images/val"
DATA_YAML= "/home/ubuntu/calorify/data/processed/uecfood100/uec_food_labels.yaml"
IMGSZ    = 640
CONF     = 0.33
IOU      = 0.60
OUTDIR   = "/home/ubuntu/calorify/results"
SAVE_VIS = True

def load_names(yaml_path: str|None, model):
    if yaml_path and Path(yaml_path).exists():
        y = yaml.safe_load(open(yaml_path, "r"))
        names = y.get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        return names
    return getattr(model, "names", None)

def draw_boxes(img, boxes_xyxy, clses, confs, names):
    out = img.copy()
    for (x1,y1,x2,y2), c, cf in zip(boxes_xyxy, clses, confs):
        color = (0, 255, 0)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cname = names[int(c)] if names and int(c) < len(names) else f"class_{int(c)}"
        label = f"{cname} {cf:.2f}"
        y = max(14, int(y1) - 6)
        cv2.putText(out, label, (int(x1), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return out

TARGET = 640

def letterbox640(img, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(TARGET / h, TARGET / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (TARGET - new_h) // 2
    bottom = TARGET - new_h - top
    left = (TARGET - new_w) // 2
    right = TARGET - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded
    
def main():
    outdir = Path(OUTDIR); (outdir/"images").mkdir(parents=True, exist_ok=True)

    if not Path(ENGINE).exists():
        print(f"[warn] Engine not found at {ENGINE}")
    if not Path(SOURCE).exists():
        print(f"[warn] Source path not found at {SOURCE}")
    if DATA_YAML and not Path(DATA_YAML).exists():
        print(f"[warn] YAML not found at {DATA_YAML}")

    model = YOLO(ENGINE)
    names = load_names(DATA_YAML, model)

    files = []
    if os.path.isdir(SOURCE):
        exts = (".jpg",".jpeg",".png",".bmp",".JPG",".PNG")
        for e in exts:
            files.extend([str(p) for p in Path(SOURCE).glob(f"*{e}")])
    else:
        files = [SOURCE]
    files = sorted(files)
    assert files, f"No images found in {SOURCE}"

    for w in files[:3]:
        img0 = cv2.imread(w)
        img640 = letterbox640(img0)
        _ = model.predict(source=img640, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)

    per_image = []
    t0 = time.perf_counter()
    for f in files:
        img0 = cv2.imread(f)
        img640 = letterbox640(img0)
        t_start = time.perf_counter()
        res = model.predict(source=img640, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)[0]
        t_end = time.perf_counter()
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.zeros((0,4))
        clses = boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros((0,),dtype=int)
        confs = boxes.conf.cpu().numpy() if boxes is not None else np.zeros((0,))
        sp = getattr(res, "speed", {})  # ms: preprocess/inference/postprocess

        per_image.append({
            "image": f,
            "preprocess_ms": float(sp.get("preprocess", np.nan)),
            "inference_ms": float(sp.get("inference", np.nan)),
            "postprocess_ms": float(sp.get("postprocess", np.nan)),
            "total_ms": (t_end - t_start) * 1000.0,
            "num_dets": int(len(confs)),
        })

        if SAVE_VIS:
            img = res.orig_img
            vis = draw_boxes(img, xyxy, clses, confs, names)
            out_path = outdir/"images"/(Path(f).stem + "_pred.jpg")
            cv2.imwrite(str(out_path), vis)

    total_time = time.perf_counter() - t0
    lat = np.array([r["total_ms"] for r in per_image], dtype=float)
    pre = np.array([r["preprocess_ms"] for r in per_image], dtype=float)
    inf = np.array([r["inference_ms"] for r in per_image], dtype=float)
    post= np.array([r["postprocess_ms"] for r in per_image], dtype=float)

    def stats(x):
        x = x[np.isfinite(x)]
        if len(x)==0: return {"mean_ms": None, "p50_ms": None, "p90_ms": None, "p95_ms": None}
        return {
            "mean_ms": float(np.mean(x)),
            "p50_ms": float(np.percentile(x, 50)),
            "p90_ms": float(np.percentile(x, 90)),
            "p95_ms": float(np.percentile(x, 95)),
        }

    summary = {
        "engine": ENGINE,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "num_images": len(files),
        "total_time_s": total_time,
        "throughput_images_per_s": len(files)/total_time if total_time>0 else None,
        "latency_total_ms": stats(lat),
        "latency_preprocess_ms": stats(pre),
        "latency_inference_ms": stats(inf),
        "latency_postprocess_ms": stats(post),
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {outdir/'summary.json'} and {outdir/'images'}")

if __name__ == "__main__":
    main()
