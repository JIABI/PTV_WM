"""Report utilities for JSON/CSV/Markdown/LaTeX artifacts and paper figures."""
from __future__ import annotations

from pathlib import Path
import csv

from ralagwm.utils.io import dump_json, ensure_dir


def save_metrics(path: str | Path, metrics: dict) -> None:
    dump_json(path, metrics)


def save_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: str | Path, title: str, metrics: dict | list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if isinstance(metrics, list):
        if metrics:
            keys = list(metrics[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "|".join(["---"] * len(keys)) + "|")
            for row in metrics:
                lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
    p.write_text("\n".join(lines) + "\n")


def save_figure(fig, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fig.savefig(p, bbox_inches="tight", dpi=160)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def save_latex_table(path: str | Path, caption: str, label: str, rows: list[dict]) -> None:
    """Write a lightweight LaTeX table for paper integration."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("% empty table\n")
        return
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{" + ("l" * len(keys)) + r"}",
        r"\toprule",
        " & ".join(keys) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    p.write_text("\n".join(lines) + "\n")
