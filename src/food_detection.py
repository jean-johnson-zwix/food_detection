#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


#ENGINE_PATH   = "/home/ubuntu/calorify/saved_models/best.onnx" 
ENGINE_PATH   = "/home/ubuntu/calorify/saved_models/best_tensorrt_fp16.engine"
SOURCE_PATH   = "/home/ubuntu/calorify/data/processed/uecfood100/images/val"  # validation images
DATA_YAML     = "/home/ubuntu/calorify/data/processed/uecfood100/uec_food_labels.yaml"
IMAGE_SIZE    = 640
CONFIDENCE    = 0.33
IOU           = 0.60

RESULTS_ROOT  = Path("/home/ubuntu/calorify/results")

def load_names(yaml_path: str | None, model):
    if yaml_path and Path(yaml_path).exists():
        y = yaml.safe_load(open(yaml_path, "r"))
        names = y.get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        return names
    return getattr(model, "names", None)


def ensure_dirs(base: Path, want_vis: bool, want_txt: bool, want_json: bool):
    base.mkdir(parents=True, exist_ok=True)
    if want_vis:
        (base / "images").mkdir(parents=True, exist_ok=True)
    if want_txt:
        (base / "labels").mkdir(parents=True, exist_ok=True)
    if want_json:
        (base / "json").mkdir(parents=True, exist_ok=True)


def save_yolo_txt(txt_path: Path, boxes_xywhn, clses, confs):
    with open(txt_path, "w") as f:
        for (x, y, w, h), c, cf in zip(boxes_xywhn, clses, confs):
            f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {cf:.6f}\n")


def draw_boxes(img, boxes_xyxy, clses, confs, names):
    out = img.copy()
    for (x1, y1, x2, y2), c, cf in zip(boxes_xyxy, clses, confs):
        color = (0, 255, 0)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cname = names[int(c)] if names and int(c) < len(names) else f"class_{int(c)}"
        label = f"{cname} {cf:.2f}"
        ytxt = max(14, int(y1) - 6)
        cv2.putText(out, label, (int(x1), ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_images(model, source: str, outdir: Path, names, imgsz: int, conf: float, iou: float):
    t0 = perf_counter()
    results = model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou, stream=True, verbose=False)
    n, file_summaries = 0, []

    for res in results:
        n += 1
        p = Path(res.path)
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            record = {"image": str(res.path), "num_dets": 0, "detections": []}
            (outdir / "json" / (p.stem + ".json")).write_text(json.dumps(record, indent=2))
            (outdir / "labels" / (p.stem + ".txt")).write_text("")  # empty file
            cv2.imwrite(str(outdir / "images" / (p.stem + "_pred.jpg")), res.orig_img)
            file_summaries.append(record)
            continue

        xyxy  = boxes.xyxy.cpu().numpy()
        xywhn = boxes.xywhn.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        vis = draw_boxes(res.orig_img, xyxy, clses, confs, names)
        cv2.imwrite(str(outdir / "images" / (p.stem + "_pred.jpg")), vis)

        save_yolo_txt(outdir / "labels" / (p.stem + ".txt"), xywhn, clses, confs)

        record = {
            "image": str(res.path),
            "num_dets": int(len(confs)),
            "detections": [
                {
                    "bbox_xyxy": [float(x) for x in b],
                    "bbox_xywhn": [float(x) for x in w],
                    "class_id": int(c),
                    "class_name": (names[int(c)] if names and int(c) < len(names) else f"class_{int(c)}"),
                    "confidence": float(cf),
                }
                for b, w, c, cf in zip(xyxy, xywhn, clses, confs)
            ],
        }
        (outdir / "json" / (p.stem + ".json")).write_text(json.dumps(record, indent=2))
        file_summaries.append(record)

        if n % 100 == 0:
            print(f"[info] processed {n} images...")

    dt = perf_counter() - t0
    summary = {
        "engine": ENGINE_PATH,
        "source": source,
        "files": len(file_summaries),
        "avg_time_per_image_s": (dt / max(1, n)),
        "total_time_s": dt,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
    }
    (outdir / "summary.json").write_text(json.dumps({"summary": summary, "results": file_summaries[:50]}, indent=2))
    print(f"[done] processed {n} file(s) in {dt:.2f}s  avg={dt/max(1,n):.4f}s/img")
    print(f"[outputs] {outdir}  (images/, labels/, json/, summary.json)")


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_ROOT / f"food_detect_{stamp}"
    ensure_dirs(outdir, want_vis=True, want_txt=True, want_json=True)

    print(f"[load] model: {ENGINE_PATH}")
    model = YOLO(ENGINE_PATH)
    names = load_names(DATA_YAML, model)
    print(f"[info] classes: {len(names) if names else 'unknown'}")
    print(f"[info] source:  {SOURCE_PATH}")
    print(f"[info] saving to: {outdir}")

    run_images(model, SOURCE_PATH, outdir, names, IMAGE_SIZE, CONFIDENCE, IOU)


if __name__ == "__main__":
    main()