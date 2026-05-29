"""
pruning.py — Pruning L1 Unstructured
=====================================
Script AUTONOME : peut être exécuté seul, sans lancer baseline.py avant.

Usage:
    python pruning.py
    python pruning.py  # ajuste PRUNING_SPARSITY dans config.py

Sortie:
    results/pruning.json    ← métriques
    saves/pruning/          ← modèle prunné sauvegardé

Comportement :
    - Si saves/baseline/ existe → réutilise le modèle (plus rapide)
    - Sinon → recharge depuis HuggingFace
    - Travaille sur une COPIE du modèle (l'original n'est pas modifié)
    - Compare automatiquement avec baseline si results/baseline.json existe
"""

import copy
import json
import torch
import torch.nn.utils.prune as prune
import numpy as np
from pathlib import Path
from datetime import datetime

import config
from model_loader import load_all, load_processor, load_model, load_eval_dataset, authenticate, detect_task
from evaluator import evaluate_model
from utils import get_model_size_mb, measure_inference_time, save_results, print_results_table

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

RESULTS_DIR   = Path("results")
SAVES_DIR     = Path("saves/pruning")
BASELINE_SAVE = Path("saves/baseline")
RESULTS_DIR.mkdir(exist_ok=True)
SAVES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_and_data():
    """Charge le modèle + processor + dataset, en réutilisant saves/baseline/ si dispo."""
    if BASELINE_SAVE.exists() and (BASELINE_SAVE / "task_info.json").exists():
        print("Réutilisation du modèle sauvegardé dans saves/baseline/")
        with open(BASELINE_SAVE / "task_info.json") as f:
            info = json.load(f)
        task = info["task"]
        authenticate()
        processor    = load_processor(str(BASELINE_SAVE), task)
        model        = load_model(str(BASELINE_SAVE), task)
        eval_dataset = load_eval_dataset(task, processor)
    else:
        print("Chargement depuis HuggingFace (saves/baseline/ introuvable)...")
        model, processor, eval_dataset, task = load_all()
    return model, processor, eval_dataset, task


def _prunable_modules(model):
    """Retourne les (module, 'weight') pour nn.Linear et Conv1D HuggingFace (GPT-2, etc.)."""
    try:
        from transformers.pytorch_utils import Conv1D as HFConv1D
    except ImportError:
        try:
            from transformers.modeling_utils import Conv1D as HFConv1D
        except ImportError:
            HFConv1D = None
    result = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            result.append((m, "weight"))
        elif HFConv1D is not None and isinstance(m, HFConv1D):
            result.append((m, "weight"))
    return result


def _apply_pruning(model, sparsity: float):
    """Pruning L1 global sur tous les layers Linear et Conv1D (GPT-2). Travaille sur une copie."""
    params = _prunable_modules(model)
    if not params:
        print("Aucun layer prunnable trouvé — pruning ignoré.")
        return model
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    for m, name in params:
        prune.remove(m, name)   # rend le pruning permanent (retire les masques)
    return model


def _actual_sparsity(model) -> float:
    """Calcule la fraction réelle de poids nuls sur tous les layers prunable."""
    total = zeros = 0
    for m, _ in _prunable_modules(model):
        total += m.weight.numel()
        zeros += (m.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def run():
    sparsity = config.PRUNING_SPARSITY
    print(f"\n{'='*55}")
    print(f"  PRUNING L1 ({sparsity*100:.0f}% sparsité) — {config.MODEL_NAME}")
    print(f"  Device  : {DEVICE}")
    print(f"  Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    model, processor, eval_dataset, task = _load_model_and_data()
    model_device = DEVICE if not hasattr(model, "hf_device_map") else DEVICE

    # Copie profonde pour ne pas altérer le modèle original
    if hasattr(model, "hf_device_map"):
        p_model = model   # device_map interdit deepcopy → pruning in-place
        print("Modèle avec device_map — pruning in-place (pas de copie).")
    else:
        p_model = copy.deepcopy(model).to(model_device)

    # Application du pruning
    print(f"Application pruning L1 global ({sparsity*100:.0f}%)...")
    p_model = _apply_pruning(p_model, sparsity)
    real_sparsity = _actual_sparsity(p_model)
    print(f"Sparsité réelle atteinte : {real_sparsity*100:.1f}%")

    # Évaluation
    print("\n Évaluation du modèle prunné...")
    metrics = evaluate_model(p_model, processor, eval_dataset, model_device, task)
    metrics["size_mb"]        = get_model_size_mb(p_model)
    metrics["inference_ms"]   = measure_inference_time(p_model, processor, model_device, task)
    metrics["sparsity_target"] = sparsity
    metrics["sparsity_actual"] = round(real_sparsity, 4)
    metrics["model_name"]     = config.MODEL_NAME
    metrics["task"]           = task
    metrics["timestamp"]      = datetime.now().isoformat()

    # Comparaison avec baseline
    results = {f"Pruning ({sparsity*100:.0f}%)": metrics}
    baseline_path = RESULTS_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        results = {"Baseline": baseline_data.get("Baseline", {}), **results}
        _print_comparison(baseline_data.get("Baseline", {}), metrics, sparsity)
    else:
        print("baseline.json absent — lance baseline.py pour la comparaison.")

    # Sauvegarde
    save_results(results, RESULTS_DIR / "pruning.json")
    try:
        p_model.save_pretrained(str(SAVES_DIR))
        processor.save_pretrained(str(SAVES_DIR))
        import json as _json
        with open(SAVES_DIR / "task_info.json", "w") as f:
            _json.dump({"task": task, "model_name": config.MODEL_NAME,
                        "sparsity": sparsity}, f)
        print(f"Modèle prunné → saves/pruning/")
    except Exception as e:
        print(f"Sauvegarde impossible : {e}")

    print_results_table(results)
    print(f"Résultats → results/pruning.json\n")
    return metrics


def _print_comparison(baseline: dict, metrics: dict, sparsity: float):
    if not baseline:
        return
    b_acc  = baseline.get("accuracy", 0)
    b_size = baseline.get("size_mb", 1)
    b_inf  = baseline.get("inference_ms", 1)
    p_acc  = metrics.get("accuracy", 0)
    p_size = metrics.get("size_mb", 1)
    p_inf  = metrics.get("inference_ms", 1)
    print(f"\n  {'Métrique':<20} {'Baseline':>12} {f'Pruning {sparsity*100:.0f}%':>12} {'Δ':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<20} {b_acc:>12.4f} {p_acc:>12.4f} {p_acc-b_acc:>+10.4f}")
    print(f"  {'Taille (MB)':<20} {b_size:>12.1f} {p_size:>12.1f} {(1-p_size/b_size)*100:>+9.1f}%")
    print(f"  {'Inférence (ms)':<20} {b_inf:>12.1f} {p_inf:>12.1f} {b_inf/p_inf:>+9.2f}×")


if __name__ == "__main__":
    run()
