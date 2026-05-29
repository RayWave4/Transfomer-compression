
# Résultats :
#     results/distillation.json   ← métriques
#     saves/distillation/         ← student sauvegardé

# Comportement :
#     - Si saves/baseline/ existe → réutilise le teacher sauvegardé (plus rapide)
#     - Sinon → recharge depuis HuggingFace
#     - Compare automatiquement avec baseline si results/baseline.json existe
#     - Non supporté pour les modèles vision/audio (ignoré proprement)

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import (
    DistilBertConfig, DistilBertForSequenceClassification,
    GPT2Config, GPT2LMHeadModel,
    AutoTokenizer,
)
from datasets import load_dataset
from tqdm import tqdm

import config
from model_loader import load_all, load_processor, load_model, load_eval_dataset, authenticate, detect_task
from evaluator import evaluate_model
from utils import get_model_size_mb, measure_inference_time, save_results, print_results_table

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

RESULTS_DIR   = Path("results")
SAVES_DIR     = Path("saves/distillation")
BASELINE_SAVE = Path("saves/baseline")
RESULTS_DIR.mkdir(exist_ok=True)
SAVES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Chargement ────────────────────────────────────────────────────────────────
def _load_teacher_and_data():
    if BASELINE_SAVE.exists() and (BASELINE_SAVE / "task_info.json").exists():
        print("  ♻️  Réutilisation du teacher sauvegardé dans saves/baseline/")
        with open(BASELINE_SAVE / "task_info.json") as f:
            info = json.load(f)
        task = info["task"]
        authenticate()
        processor    = load_processor(str(BASELINE_SAVE), task)
        teacher      = load_model(str(BASELINE_SAVE), task)
        eval_dataset = load_eval_dataset(task, processor)
    else:
        print("  🌐 Chargement depuis HuggingFace (saves/baseline/ introuvable)...")
        teacher, processor, eval_dataset, task = load_all()
    return teacher, processor, eval_dataset, task


# ── Builders student ──────────────────────────────────────────────────────────
def _build_classification_student(num_labels: int):
    cfg = DistilBertConfig(
        vocab_size=30522,
        max_position_embeddings=512,
        n_layers=config.DISTIL_STUDENT_LAYERS,
        n_heads=4,
        dim=config.DISTIL_STUDENT_DIM,
        hidden_dim=config.DISTIL_STUDENT_DIM * 4,
        num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(cfg)


def _build_causal_student(vocab_size: int):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=config.DISTIL_STUDENT_DIM,
        n_layer=config.DISTIL_STUDENT_LAYERS,
        n_head=4,
    )
    return GPT2LMHeadModel(cfg)


# ── Loss ──────────────────────────────────────────────────────────────────────
def _distil_loss(s_logits, t_logits, labels, T, alpha):
    ce = F.cross_entropy(s_logits, labels) if labels is not None else torch.tensor(0.0, device=s_logits.device)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)
    return alpha * ce + (1 - alpha) * kl


# ── Dataset d'entraînement ────────────────────────────────────────────────────
_TRAIN_SOURCES = {
    "text-classification":  ("glue",     "sst2",                 "train"),
    "token-classification": ("conll2003", None,                   "train"),
    "text-generation":      ("Salesforce/wikitext", "wikitext-2-raw-v1", "train"),
    "text2text-generation": ("EdinburghNLP/xsum",   None,                "train[:10%]"),
    "question-answering":   ("rajpurkar/squad",     None,                "train[:5%]"),
}

def _load_train_loader(task: str, processor):
    path, name, split = _TRAIN_SOURCES.get(task, ("stanfordnlp/imdb", "sst2", "train"))
    raw = load_dataset(path, name, split=split, token=config.HF_TOKEN or True)
    if config.DISTIL_TRAIN_SAMPLES:
        raw = raw.select(range(min(config.DISTIL_TRAIN_SAMPLES, len(raw))))

    text_col = next((c for c in ["sentence", "text", "document", "question", "tokens"]
                     if c in raw.column_names), raw.column_names[0])
    label_col = next((c for c in ["label", "labels", "ner_tags"] if c in raw.column_names), None)

    def tok(batch):
        texts = batch[text_col]
        if isinstance(texts[0], list):
            texts = [" ".join(t) for t in texts]
        return processor(texts, truncation=True, padding="max_length", max_length=config.MAX_LENGTH)

    ds = raw.map(tok, batched=True)
    keep = [c for c in ["input_ids", "attention_mask", label_col] if c and c in ds.column_names]
    ds.set_format(type="torch", columns=keep)
    if label_col and label_col != "label" and label_col in ds.column_names:
        ds = ds.rename_column(label_col, "label")
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)


# ── Entraînement ──────────────────────────────────────────────────────────────
def _train(student, teacher, train_loader, task: str):
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.DISTIL_LR)
    teacher_device = next(iter(teacher.parameters())).device

    for epoch in range(1, config.DISTIL_EPOCHS + 1):
        student.train()
        teacher.eval()
        total, n = 0.0, 0

        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{config.DISTIL_EPOCHS}", leave=False):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            lbls  = batch.get("label", batch.get("labels"))
            if lbls is not None:
                lbls = lbls.to(DEVICE)

            with torch.no_grad():
                t_logits = teacher(input_ids=ids.to(teacher_device),
                                   attention_mask=mask.to(teacher_device)).logits.to(DEVICE)

            s_logits = student(input_ids=ids, attention_mask=mask).logits

            # Aligner shapes pour les modèles génératifs
            if s_logits.dim() == 3:
                s_logits = s_logits[:, :-1].contiguous().view(-1, s_logits.size(-1))
                t_logits = t_logits[:, :-1].contiguous().view(-1, t_logits.size(-1))
                lbls = ids[:, 1:].contiguous().view(-1) if lbls is None else None

            loss = _distil_loss(s_logits, t_logits, lbls, config.DISTIL_TEMPERATURE, config.DISTIL_ALPHA)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1

        print(f"  Epoch {epoch} — loss moy : {total/n:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print(f"\n{'='*55}")
    print(f"  KNOWLEDGE DISTILLATION — {config.MODEL_NAME}")
    print(f"  Device  : {DEVICE}")
    print(f"  Démarré : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    teacher, processor, eval_dataset, task = _load_teacher_and_data()

    if task in ("image-classification", "audio-classification"):
        print("  ⚠️  Distillation non supportée pour vision/audio — ignorée.")
        metrics = {"skipped": True, "reason": f"task={task} non supportée"}
        save_results({"Distillation": metrics}, RESULTS_DIR / "distillation.json")
        return metrics

    # Construire le student
    num_labels  = getattr(teacher.config, "num_labels", 2)
    vocab_size  = getattr(processor, "vocab_size", 30522)
    student = (_build_causal_student(vocab_size) if task == "text-generation"
               else _build_classification_student(num_labels))
    student.to(DEVICE)

    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  Teacher : {t_params:.1f}M params")
    print(f"  Student : {s_params:.1f}M params\n")

    # Entraînement
    print("  🎓 Chargement des données d'entraînement...")
    try:
        train_loader = _load_train_loader(task, processor)
    except Exception as e:
        print(f"  ⚠️  Impossible de charger le dataset d'entraînement : {e}")
        metrics = {"skipped": True, "reason": str(e)}
        save_results({"Distillation": metrics}, RESULTS_DIR / "distillation.json")
        return metrics

    print("  🎓 Entraînement par distillation...")
    _train(student, teacher, train_loader, task)

    # Évaluation
    student.eval()
    print("\n📊 Évaluation du student...")
    metrics = evaluate_model(student, processor, eval_dataset, DEVICE, task)
    metrics["size_mb"]       = get_model_size_mb(student)
    metrics["inference_ms"]  = measure_inference_time(student, processor, DEVICE, task)
    metrics["student_params"] = round(s_params, 2)
    metrics["teacher_params"] = round(t_params, 2)
    metrics["model_name"]    = config.MODEL_NAME
    metrics["task"]          = task
    metrics["timestamp"]     = datetime.now().isoformat()

    # Comparaison avec baseline
    results = {"Distillation": metrics}
    baseline_path = RESULTS_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        results = {"Baseline": baseline_data.get("Baseline", {}), **results}
        _print_comparison(baseline_data.get("Baseline", {}), metrics)
    else:
        print("  ℹ️  baseline.json absent — lance baseline.py pour la comparaison.")

    # Sauvegarde
    save_results(results, RESULTS_DIR / "distillation.json")
    student.save_pretrained(str(SAVES_DIR))
    processor.save_pretrained(str(SAVES_DIR))
    with open(SAVES_DIR / "task_info.json", "w") as f:
        json.dump({"task": task, "model_name": config.MODEL_NAME, "is_student": True}, f)
    print(f"  💾 Student → saves/distillation/")

    print_results_table(results)
    print(f"✅ Résultats → results/distillation.json\n")
    return metrics


def _print_comparison(baseline: dict, metrics: dict):
    if not baseline:
        return
    b_acc  = baseline.get("accuracy", 0)
    b_size = baseline.get("size_mb", 1)
    b_inf  = baseline.get("inference_ms", 1)
    d_acc  = metrics.get("accuracy", 0)
    d_size = metrics.get("size_mb", 1)
    d_inf  = metrics.get("inference_ms", 1)
    print(f"\n  {'Métrique':<20} {'Baseline':>12} {'Student':>12} {'Δ':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<20} {b_acc:>12.4f} {d_acc:>12.4f} {d_acc-b_acc:>+10.4f}")
    print(f"  {'Taille (MB)':<20} {b_size:>12.1f} {d_size:>12.1f} {(1-d_size/b_size)*100:>+9.1f}%")
    print(f"  {'Inférence (ms)':<20} {b_inf:>12.1f} {d_inf:>12.1f} {b_inf/d_inf:>+9.2f}×")


if __name__ == "__main__":
    run()
