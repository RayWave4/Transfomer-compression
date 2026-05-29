"""
baseline.py — Chargement et évaluation du modèle original (non compressé)
==========================================================================
Script AUTONOME : peut être exécuté seul.

Usage:
    python baseline.py

Sortie:
    results/baseline.json   ← métriques sauvegardées
    saves/baseline/         ← modèle + processor sauvegardés pour réutilisation
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

import config
from model_loader import load_all
from evaluator import evaluate_model
from utils import get_model_size_mb, measure_inference_time, save_results, print_results_table

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

RESULTS_DIR = Path("results")
SAVES_DIR   = Path("saves/baseline")
RESULTS_DIR.mkdir(exist_ok=True)
SAVES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print(f"\n{'='*55}")
    print(f"  BASELINE — {config.MODEL_NAME}")
    print(f"  Device  : {DEVICE}")
    print(f"  Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    # Chargement complet (authentification + détection tâche + dataset)
    model, processor, eval_dataset, task = load_all()
    model_device = DEVICE if not hasattr(model, "hf_device_map") else DEVICE

    # Évaluation
    print("📊 Évaluation du baseline...")
    metrics = evaluate_model(model, processor, eval_dataset, model_device, task)
    metrics["size_mb"]      = get_model_size_mb(model)
    metrics["inference_ms"] = measure_inference_time(model, processor, model_device, task)
    metrics["model_name"]   = config.MODEL_NAME
    metrics["task"]         = task
    metrics["timestamp"]    = datetime.now().isoformat()

    # Sauvegarde des résultats
    save_results({"Baseline": metrics}, RESULTS_DIR / "baseline.json")

    # Sauvegarde du modèle + processor pour réutilisation par les autres scripts
    print(f"\n💾 Sauvegarde du modèle dans {SAVES_DIR}/...")
    try:
        model.save_pretrained(str(SAVES_DIR))
        processor.save_pretrained(str(SAVES_DIR))
        # Sauvegarder aussi la tâche détectée pour les scripts suivants
        import json
        with open(SAVES_DIR / "task_info.json", "w") as f:
            json.dump({"task": task, "model_name": config.MODEL_NAME}, f)
        print("  ✓ Modèle sauvegardé.")
    except Exception as e:
        print(f"  ⚠️  Sauvegarde impossible ({e}) — les autres scripts rechargeront depuis HF.")

    # Affichage
    print_results_table({"Baseline": metrics})
    print(f"✅ Résultats → results/baseline.json")
    print(f"   Lance ensuite : python quantization.py | python pruning.py | python distillation.py\n")
    return metrics


if __name__ == "__main__":
    run()
