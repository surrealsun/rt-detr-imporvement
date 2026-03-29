from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into an ablation table.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Directories containing metrics.json files.")
    parser.add_argument("--output", default="reports/ablation_summary.md", help="Markdown summary path.")
    parser.add_argument("--csv", default="reports/ablation_summary.csv", help="CSV summary path.")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    rows = []
    for run_dir in [Path(path) for path in args.run_dirs]:
        metrics_path = run_dir / "eval" / "metrics.json"
        if not metrics_path.exists():
            metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        rows.append(
            {
                "experiment": run_dir.name,
                "mAP": metrics.get("mAP", 0.0),
                "AP50": metrics.get("AP50", 0.0),
                "APsmall": metrics.get("APsmall", 0.0),
                "fps": metrics.get("fps", 0.0),
                "params_m": metrics.get("params", 0.0) / 1e6,
                "flops_g": metrics.get("flops", 0.0) / 1e9 if metrics.get("flops") is not None else None,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("| Experiment | mAP | AP50 | APsmall | FPS | Params (M) | FLOPs (G) |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            flops_g = row["flops_g"]
            flops_text = f"{flops_g:.2f}" if flops_g is not None else "n/a"
            handle.write(
                f"| {row['experiment']} | {row['mAP']:.4f} | {row['AP50']:.4f} | "
                f"{row['APsmall']:.4f} | {row['fps']:.2f} | {row['params_m']:.2f} | {flops_text} |\n"
            )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["experiment"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
