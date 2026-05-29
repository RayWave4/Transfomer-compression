"""
utils.py — Utilitaires universels : taille, timing, affichage, sauvegarde
"""

import io
import json
import time
import torch
import numpy as np
from pathlib import Path
import config


# ── Taille du modèle ──────────────────────────────────────────────────────────
def get_model_size_mb(model) -> float:
    buf = io.BytesIO()
    try:
        torch.save(model.state_dict(), buf)
        return buf.tell() / (1024 ** 2)
    except Exception:
        # Fallback : compter les paramètres × 4 octets (float32)
        n_params = sum(p.numel() for p in model.parameters())
        return n_params * 4 / (1024 ** 2)


# ── Temps d'inférence ─────────────────────────────────────────────────────────
def measure_inference_time(model, processor, device, task: str, n_runs: int = 30) -> float:
    """
    Mesure le temps d'inférence moyen (ms) sur un input fictif adapté à la tâche.
    """
    model.eval()
    target_device = device

    # Construire un input de test selon le type de tâche
    try:
        if task in ("text-classification", "token-classification",
                    "text-generation", "text2text-generation", "question-answering"):
            dummy = processor(
                "This is a benchmark sentence for measuring inference time.",
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=config.MAX_LENGTH,
            )
            inputs = {k: v.to(target_device) for k, v in dummy.items()
                      if k in ("input_ids", "attention_mask")}

        elif task == "image-classification":
            from PIL import Image
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            dummy = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(target_device) for k, v in dummy.items()}

        elif task == "audio-classification":
            audio = np.zeros(16000, dtype=np.float32)  # 1 sec silence
            dummy = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(target_device) for k, v in dummy.items()}

        else:
            return 0.0

    except Exception as e:
        print(f"  ⚠️  Impossible de créer l'input de test ({e}) — timing=0")
        return 0.0

    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            try:
                model(**inputs)
            except Exception:
                return 0.0

    # Mesure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(**inputs)
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ── Sauvegarde / Chargement ───────────────────────────────────────────────────
def save_results(results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Affichage du tableau ──────────────────────────────────────────────────────
def print_results_table(results: dict):
    baseline = results.get("Baseline", {})
    base_size = baseline.get("size_mb", 1)
    base_inf  = baseline.get("inference_ms", 1)
    base_acc  = baseline.get("accuracy", baseline.get("perplexity", 0))

    metric_name = "Accuracy" if "accuracy" in baseline else "Perplexity"

    header = (f"\n{'Méthode':<30} {metric_name:>12} {'Taille (MB)':>12} "
              f"{'Inférence (ms)':>16} {'Compression':>12} {'Speedup':>10}")
    print(header)
    print("─" * len(header))

    for name, m in results.items():
        if m.get("skipped"):
            print(f"  {name:<28} {'[ignoré]':>12}")
            continue

        metric_val = m.get("accuracy", m.get("perplexity", 0))
        size       = m.get("size_mb", 0)
        inf_ms     = m.get("inference_ms", 0)
        compression = base_size / size if size > 0 else 0
        speedup     = base_inf / inf_ms if inf_ms > 0 else 0

        print(
            f"  {name:<28} "
            f"{metric_val:>12.3f} "
            f"{size:>12.1f} "
            f"{inf_ms:>16.1f} "
            f"{compression:>11.2f}× "
            f"{speedup:>9.2f}×"
        )
    print()
