#!/usr/bin/env python

import os, json, glob, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


BASE_RESULTS_DIR = Path("/home/ubuntu/calorify/results")
RUN_PREFIX       = "food_detect_"


def load_all_summaries(base_dir: Path, prefix: str = RUN_PREFIX):
    rows = []
    for run_dir in sorted(base_dir.glob(f"{prefix}*")):
        sfile = run_dir / "summary.json"
        if not sfile.exists():
            continue
        try:
            data = json.loads(sfile.read_text())
            meta = data.get("summary", {})
            engine_path = str(meta.get("engine", "")) if meta else ""
            engine_base = os.path.basename(engine_path) if engine_path else ""
            if engine_path.endswith(".engine"):
                engine_type = "TensorRT (FP16)"
            elif engine_path.endswith(".onnx"):
                engine_type = "ONNX Runtime"
            else:
                engine_type = "Unknown"

            avg_s = float(meta.get("avg_time_per_image_s", float("nan")))
            files = int(meta.get("files", 0))
            imgsz = meta.get("imgsz", None)
            conf  = meta.get("conf", None)
            iou   = meta.get("iou", None)
            source= meta.get("source", "")
            total = float(meta.get("total_time_s", float("nan")))

            rows.append(dict(
                run_name=f'{engine_type} run',
                run_path=str(run_dir),
                engine_type=engine_type,
                engine_path=engine_path,
                engine_file=engine_base,
                files=files,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                avg_ms=avg_s * 1000.0 if math.isfinite(avg_s) else float("nan"),
                throughput_ips=(1.0 / avg_s) if avg_s and avg_s > 0 else float("nan"),
                total_time_s=total,
                source=source
            ))
        except Exception as e:
            print(f"[warn] could not parse {sfile}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["engine_type", "avg_ms"], ascending=[True, True]).reset_index(drop=True)
    return df


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def fmt_ms(x):
    return "—" if not math.isfinite(x) else f"{x:.2f} ms"


def fmt_ips(x):
    return "—" if not math.isfinite(x) else f"{x:.1f} img/s"


def make_charts(df: pd.DataFrame, outdir: Path):
    if df.empty:
        print("[info] no runs found; skipping charts.")
        return

    # Chart 1: Average Latency (ms)
    fig1, ax1 = plt.subplots(figsize=(10, max(3, 0.45 * len(df))))
    y = range(len(df))
    ax1.barh(y, df["avg_ms"].values)  # default color; no custom styles
    ax1.set_yticks(y, labels=df["run_name"])
    ax1.set_xlabel("Avg latency per image (ms)")
    ax1.set_title("Food Detection — Avg Latency (lower is better)")
    for i, v in enumerate(df["avg_ms"].values):
        ax1.text(v, i, f" {fmt_ms(v)}", va="center")
    fig1.tight_layout()
    p1 = outdir / "latency_ms_bar.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # Chart 2: Throughput (images/s)
    fig2, ax2 = plt.subplots(figsize=(10, max(3, 0.45 * len(df))))
    ax2.barh(y, df["throughput_ips"].values)
    ax2.set_yticks(y, labels=df["run_name"])
    ax2.set_xlabel("Throughput (images/s)")
    ax2.set_title("Food Detection — Throughput (higher is better)")
    for i, v in enumerate(df["throughput_ips"].values):
        ax2.text(v, i, f" {fmt_ips(v)}", va="center")
    fig2.tight_layout()
    p2 = outdir / "throughput_ips_bar.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    print(f"[charts] wrote:\n - {p1}\n - {p2}")


def write_tables(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    table_cols = [
        "run_name", "engine_type", "engine_file", "files", "imgsz",
        "conf", "iou", "avg_ms", "throughput_ips", "total_time_s"
    ]
    tdf = df[table_cols].copy()
    tdf["avg_ms"] = tdf["avg_ms"].map(lambda x: None if not math.isfinite(x) else round(x, 3))
    tdf["throughput_ips"] = tdf["throughput_ips"].map(lambda x: None if not math.isfinite(x) else round(x, 2))
    tdf["total_time_s"] = tdf["total_time_s"].map(lambda x: None if not math.isfinite(x) else round(x, 3))

    # CSV + Markdown
    csv_path = outdir / "runs_summary.csv"
    md_path  = outdir / "runs_summary.md"
    tdf.to_csv(csv_path, index=False)

    md = ["# Food Detection (ONNX vs FP16) — Run Summary", "", tdf.to_markdown(index=False), ""]
    md_path.write_text("\n".join(md))
    print(f"[tables] wrote:\n - {csv_path}\n - {md_path}")


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = BASE_RESULTS_DIR / f"comparitive_analysis_results_{stamp}"
    ensure_dir(outdir)

    df = load_all_summaries(BASE_RESULTS_DIR, RUN_PREFIX)
    if df.empty:
        print(f"[info] No 'summary.json' files found under {BASE_RESULTS_DIR}/{RUN_PREFIX}*/")
        return

    print(df[["run_name","engine_type","engine_file","files","avg_ms","throughput_ips"]])

    write_tables(df, outdir)
    make_charts(df, outdir)
    print(f"[done] outputs at: {outdir}")


if __name__ == "__main__":
    main()
