#!/usr/bin/env python3
import random, shutil
from pathlib import Path
import cv2
import yaml
import kagglehub

def _path(p):
    return p if isinstance(p, Path) else Path(p)

def copy_dataset(source, folder_name):
    raw_data_path = Path('data')/ 'raw'
    destination = raw_data_path / folder_name
    if destination.exists():
        print(f"{folder_name} already exists, skipping copy...")
        return
    try:
        shutil.copytree(source, destination)
        print(f"Copied {folder_name} successfully")
    except Exception as e:
        print(f"Error copying {folder_name}: {e}")

def download_dataset():
    print("Starting dataset download")
    uecfood100_dataset = kagglehub.dataset_download('rkuo2000/uecfood100')
    print(f"Downloaded UEC Food 100 dataset to {uecfood100_dataset}")
    copy_dataset(uecfood100_dataset,'uecfood100')

def read_category_map(uec_root: Path):
    cat_file = _path(uec_root) / "category.txt"
    id2name = {}
    if cat_file.exists():
        for line in cat_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            try:
                cid = int(parts[0])
                name = " ".join(parts[1:]) if len(parts) > 1 else str(cid)
                id2name[cid] = name
            except:
                continue
    return id2name

def list_class_dirs(uec_root: Path):
    uec_root = _path(uec_root)
    return sorted([p for p in uec_root.iterdir() if p.is_dir() and p.name.isdigit()],
                  key=lambda p: int(p.name))

def load_bb_info(class_dir: Path):
    bb = {}
    class_dir = _path(class_dir)
    info = class_dir / "bb_info.txt"
    if not info.exists():
        return bb
    for line in info.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) < 5: 
            continue
        fname = parts[0]
        try:
            x1, y1, x2, y2 = map(int, parts[1:5])
        except:
            continue
        img_path = class_dir / fname
        if not img_path.exists():
            base = Path(fname).stem
            hit = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
                cand = class_dir / f"{base}{ext}"
                if cand.exists():
                    hit = cand; break
            if hit is None:
                for p in class_dir.iterdir():
                    if p.is_file() and p.stem == base:
                        hit = p; break
            if hit is None:
                continue
            img_path = hit
        bb.setdefault(img_path, []).append((x1, y1, x2, y2))
    return bb

def xyxy_to_yolo(x1,y1,x2,y2,w,h):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0
    return cx/w, cy/h, bw/w, bh/h

def main():

    download_dataset()

    RAW_DATA_PATH = 'data/raw/uecfood100'
    PROCESSED_DATA_PATH = 'data/processed/uecfood100'
    VAL_RATIO = 0.1
    SEED = 42
    MAX_CLASSES=0
    IMG_EXTENSION = [".jpg", ".jpeg", ".png", ".bmp"]
    YAML_FILE_NAME = 'uec_food_labels.yaml'

    random.seed(SEED)

    RAW_DATA_PATH = _path(RAW_DATA_PATH)
    PROCESSED_DATA_PATH = _path(PROCESSED_DATA_PATH)
    if not (RAW_DATA_PATH / "category.txt").exists():
        subs = [d for d in RAW_DATA_PATH.iterdir() if d.is_dir()]
        for d in subs:
            if (d / "category.txt").exists():
                print(f"[info] Auto-detected dataset root at {d}")
                RAW_DATA_PATH = d
                break

    assert RAW_DATA_PATH.exists(), f"Path does not exist: {RAW_DATA_PATH}"
    assert (RAW_DATA_PATH / "category.txt").exists(), (
        f"Could not find category.txt under {RAW_DATA_PATH}. "
        "Point RAW_DATA_PATH to the folder that contains numbered class dirs and category.txt."
    )

    class_dirs = list_class_dirs(Path(RAW_DATA_PATH))
    if MAX_CLASSES and MAX_CLASSES > 0:
        class_dirs = class_dirs[:MAX_CLASSES]

    id2name = read_category_map(RAW_DATA_PATH)
    folder_ids = [int(d.name) for d in class_dirs]
    id2idx = {fid: i for i, fid in enumerate(folder_ids)}
    names = [id2name.get(fid, str(fid)) for fid in folder_ids]

    samples = []
    per_image = {}
    for d in class_dirs:
        fid = int(d.name)
        cidx = id2idx[fid]
        bb = load_bb_info(d)
        for img_path, boxes in bb.items():
            key = (fid, img_path.stem)
            entry = per_image.setdefault(key, {"img_path": img_path, "class_idx": cidx, "boxes": []})
            entry["boxes"].extend(boxes)

    items = list(per_image.values())
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    val_set = set(id(entry) for entry in items[:n_val])

    for split in ["train", "val"]:
        (PROCESSED_DATA_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_count = {"train":0, "val":0}
    ann_count = {"train":0, "val":0}
    for entry in items:
        split = "val" if id(entry) in val_set else "train"
        src = entry["img_path"]
        cidx = entry["class_idx"]

        dst_stem = f"{folder_ids[cidx]}_{src.stem}"
        dst_img = PROCESSED_DATA_PATH / "images" / split / (dst_stem + src.suffix.lower())
        dst_lbl = PROCESSED_DATA_PATH / "labels" / split / (dst_stem + ".txt")

        img = cv2.imread(str(src))
        if img is None: 
            continue
        h, w = img.shape[:2]

        lines = []
        for (x1,y1,x2,y2) in entry["boxes"]:
            cx, cy, bw, bh = xyxy_to_yolo(x1,y1,x2,y2,w,h)
            if bw <= 0 or bh <= 0: 
                continue
            lines.append(f"{cidx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        shutil.copy2(src, dst_img)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")

        img_count[split] += 1
        ann_count[split] += len(lines)

    yaml_path = PROCESSED_DATA_PATH / YAML_FILE_NAME
    cfg = {
        "path": str(PROCESSED_DATA_PATH),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)}
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Done. Output @ {PROCESSED_DATA_PATH}")
    print(f"Train images: {img_count['train']} | Labels: {ann_count['train']}")
    print(f"Val   images: {img_count['val']} | Labels: {ann_count['val']}")
    print(f"YAML : {yaml_path}")

if __name__ == "__main__":
    main()
