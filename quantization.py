"""
quantization.py — Quantization INT8 (Dynamic ou BitsAndBytes)
==============================================================
Script AUTONOME : peut être exécuté seul, sans lancer baseline.py avant.

Usage:
    python quantization.py

Sortie:
    results/quantization.json   ← métriques
    saves/quantization/         ← modèle quantizé sauvegardé

Comportement :
    - Si saves/baseline/ existe → réutilise le modèle sauvegardé (plus rapide)
    - Sinon → recharge le modèle depuis HuggingFace directement
    - Compare automatiquement avec baseline si results/baseline.json existe
"""

import copy
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.quantization import quantize_dynamic

import config
from model_loader import load_all, load_processor, load_model, load_eval_dataset, authenticate, detect_task
from evaluator import evaluate_model
from utils import get_model_size_mb, measure_inference_time, save_results, print_results_table

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

RESULTS_DIR = Path("results")
SAVES_DIR   = Path("saves/quantization")
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
        processor = load_processor(str(BASELINE_SAVE), task)
        model     = load_model(str(BASELINE_SAVE), task)
        eval_dataset = load_eval_dataset(task, processor)
    else:
        print("Chargement depuis HuggingFace (saves/baseline/ introuvable)...")
        model, processor, eval_dataset, task = load_all()
    return model, processor, eval_dataset, task


def run():
    print(f"\n{'='*55}")
    print(f"  QUANTIZATION (INT8) — {config.MODEL_NAME}")
    print(f"  Device  : {DEVICE}")
    print(f"  Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    model, processor, eval_dataset, task = _load_model_and_data()
    model_device = DEVICE if not hasattr(model, "hf_device_map") else DEVICE

    # ── Choisir la méthode selon le contexte ─────────────────────────────────
    if task in ("text-generation", "text2text-generation") and DEVICE.type == "cuda":
        try:
            q_model, method = _quantize_bnb(task), "BitsAndBytes 8-bit"
        except ImportError:
            print(" bitsandbytes absent → Dynamic INT8 (CPU)")
            q_model, method = _quantize_dynamic(model), "Dynamic INT8"
            model_device = torch.device("cpu")
    else:
        q_model, method = _quantize_dynamic(model), "Dynamic INT8"
        model_device = torch.device("cpu")   # dynamic quant = CPU uniquement

    # ── Évaluation ────────────────────────────────────────────────────────────
    print(f"\n Évaluation ({method})...")
    metrics = evaluate_model(q_model, processor, eval_dataset, model_device, task)
    metrics["size_mb"]      = get_model_size_mb(q_model)
    metrics["inference_ms"] = measure_inference_time(q_model, processor, model_device, task)
    metrics["method"]       = method
    metrics["model_name"]   = config.MODEL_NAME
    metrics["task"]         = task
    metrics["timestamp"]    = datetime.now().isoformat()

    # ── Comparaison avec baseline ──────────────────────────────────────────────
    results = {"Quantization": metrics}
    baseline_path = RESULTS_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        results = {"Baseline": baseline_data.get("Baseline", {}), **results}
        _print_comparison(baseline_data.get("Baseline", {}), metrics)
    else:
        print("baseline.json absent — lance baseline.py pour la comparaison.")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    save_results(results, RESULTS_DIR / "quantization.json")
    try:
        torch.save(q_model.state_dict(), SAVES_DIR / "model_int8.pt")
        print(f"Modèle quantizé → saves/quantization/model_int8.pt")
    except Exception as e:
        print(f"Sauvegarde state_dict impossible : {e}")

    print_results_table(results)
    print(f"Résultats → results/quantization.json\n")
    return metrics


def _quantize_dynamic(model):
    """Dynamic quantization INT8 sur tous les layers Linear (CPU)."""
    print("Application Dynamic Quantization INT8...")
    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    q = quantize_dynamic(model_cpu, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8)
    q.eval()
    return q


def _quantize_bnb(task: str):
    """Quantization 8-bit via bitsandbytes (GPU)."""
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    print("Application BitsAndBytes 8-bit (GPU)...")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    cls = AutoModelForCausalLM if task == "text-generation" else AutoModelForSeq2SeqLM
    q = cls.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_cfg,
        token=config.HF_TOKEN or True,
        device_map="auto",
    )
    q.eval()
    return q


def _print_comparison(baseline: dict, metrics: dict):
    if not baseline:
        return
    b_acc  = baseline.get("accuracy", 0)
    b_size = baseline.get("size_mb", 1)
    b_inf  = baseline.get("inference_ms", 1)
    q_acc  = metrics.get("accuracy", 0)
    q_size = metrics.get("size_mb", 1)
    q_inf  = metrics.get("inference_ms", 1)
    print(f"\n  {'Métrique':<20} {'Baseline':>12} {'Quantizé':>12} {'Δ':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<20} {b_acc:>12.4f} {q_acc:>12.4f} {q_acc-b_acc:>+10.4f}")
    print(f"  {'Taille (MB)':<20} {b_size:>12.1f} {q_size:>12.1f} {(1-q_size/b_size)*100:>+9.1f}%")
    print(f"  {'Inférence (ms)':<20} {b_inf:>12.1f} {q_inf:>12.1f} {b_inf/q_inf:>+9.2f}×")


if __name__ == "__main__":
    run()
