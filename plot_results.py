"""
plot_results.py — Génération des graphiques depuis results/results.json
Usage: python plot_results.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_PATH = Path("results/results.json")
OUTPUT_DIR   = Path("results")

PALETTE = ["#2D6A9F", "#E07B39", "#3BAF6E", "#C0392B", "#8E44AD", "#16A085"]


def load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def main():
    results = load()
    names   = [n for n, m in results.items() if not m.get("skipped")]
    colors  = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

    # Détecter la métrique principale
    metric_key = "accuracy" if "accuracy" in results["Baseline"] else "perplexity"
    metric_label = "Accuracy" if metric_key == "accuracy" else "Perplexité"

    metric_vals  = [results[n][metric_key]     for n in names]
    size_vals    = [results[n]["size_mb"]       for n in names]
    inf_vals     = [results[n]["inference_ms"]  for n in names]

    base_size = results["Baseline"]["size_mb"]
    base_inf  = results["Baseline"]["inference_ms"]
    comp_vals = [base_size / results[n]["size_mb"]      for n in names]
    speed_vals= [base_inf  / results[n]["inference_ms"] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Compression de Transformers — Comparaison des méthodes",
                 fontsize=15, fontweight="bold", y=1.01)

    def bar(ax, vals, ylabel, title, fmt):
        bars = ax.bar(names, vals, color=[colors[n] for n in names],
                      edgecolor="white", linewidth=1.3)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                    f"{v:{fmt}}", ha="center", fontsize=9, fontweight="bold")

    bar(axes[0, 0], metric_vals,  metric_label,       metric_label + " par méthode",     ".3f")
    bar(axes[0, 1], size_vals,    "Taille (MB)",       "Taille du modèle (MB)",           ".1f")
    bar(axes[1, 0], inf_vals,     "Temps (ms)",        "Temps d'inférence (ms)",          ".1f")
    bar(axes[1, 1], comp_vals,    "Ratio (×)",         "Taux de compression vs. Baseline",",.2f")

    plt.tight_layout()
    out1 = OUTPUT_DIR / "comparison_charts.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"  ✓ Graphiques : {out1}")

    # Scatter : métrique principale vs. taille
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name in names:
        m = results[name]
        ax2.scatter(m["size_mb"], m[metric_key], color=colors[name],
                    s=200, zorder=5, edgecolors="white", linewidths=1.5)
        ax2.annotate(name, (m["size_mb"], m[metric_key]),
                     textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax2.set_xlabel("Taille du modèle (MB)", fontsize=11)
    ax2.set_ylabel(metric_label, fontsize=11)
    ax2.set_title(f"{metric_label} vs. Taille du modèle", fontsize=13, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "accuracy_vs_size.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  ✓ Scatter    : {out2}")
    plt.show()


if __name__ == "__main__":
    main()
