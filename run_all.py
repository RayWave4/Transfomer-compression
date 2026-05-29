"""
run_all.py — Lanceur optionnel qui exécute les 4 scripts en séquence
=====================================================================
Ce fichier est OPTIONNEL. Chaque script peut être lancé individuellement.

Usage:
    python run_all.py               # tout lancer
    python run_all.py --skip distillation
    python run_all.py --only quantization pruning
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

import config
from utils import load_results, print_results_table, save_results

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ALL_STEPS = ["baseline", "quantization", "pruning", "distillation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], choices=ALL_STEPS,
                        help="Étapes à ignorer")
    parser.add_argument("--only", nargs="*", default=None, choices=ALL_STEPS,
                        help="Étapes à exécuter uniquement")
    args = parser.parse_args()

    steps = args.only if args.only else [s for s in ALL_STEPS if s not in args.skip]

    print(f"\n{'='*62}")
    print(f"  Transformer Compression — run_all.py")
    print(f"  Modèle  : {config.MODEL_NAME}")
    print(f"  Étapes  : {' → '.join(steps)}")
    print(f"  Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}\n")

    all_results = {}

    if "baseline" in steps:
        from baseline import run as run_baseline
        m = run_baseline()
        all_results["Baseline"] = m

    if "quantization" in steps:
        from quantization import run as run_quantization
        m = run_quantization()
        all_results["Quantization (INT8)"] = m

    if "pruning" in steps:
        from pruning import run as run_pruning
        m = run_pruning()
        all_results[f"Pruning ({config.PRUNING_SPARSITY*100:.0f}%)"] = m

    if "distillation" in steps:
        from distillation import run as run_distillation
        m = run_distillation()
        all_results["Knowledge Distillation"] = m

    # Tableau récapitulatif global
    if len(all_results) > 1:
        print(f"\n{'='*62}")
        print("  RÉCAPITULATIF GLOBAL")
        print(f"{'='*62}")
        print_results_table(all_results)
        save_results(all_results, RESULTS_DIR / "all_results.json")
        print(f"✅ Récapitulatif → results/all_results.json")
        print(f"   Génère les graphiques : python plot_results.py\n")


if __name__ == "__main__":
    main()
