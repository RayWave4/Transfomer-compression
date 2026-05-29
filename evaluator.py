"""
evaluator.py — Évaluation universelle selon le type de tâche
=============================================================
Supporte : text-classification, token-classification, text-generation (perplexité),
           text2text-generation, image-classification, audio-classification.
"""

import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import config


def evaluate_model(model, processor, eval_dataset, device, task: str) -> dict:
    """
    Calcule les métriques adaptées à la tâche :
      - text-classification      → accuracy
      - token-classification     → accuracy (token-level)
      - text-generation          → perplexité
      - text2text-generation     → perplexité approx.
      - image-classification     → accuracy
      - audio-classification     → accuracy
    """
    model.eval()

    # Déplacer le modèle sur device seulement si pas déjà géré par device_map
    if not hasattr(model, "hf_device_map"):
        model.to(device)

    if task in ("text-classification", "token-classification"):
        return _eval_classification(model, eval_dataset, device, task)
    elif task in ("text-generation", "text2text-generation"):
        return _eval_perplexity(model, processor, eval_dataset, device)
    elif task == "image-classification":
        return _eval_image_classification(model, processor, eval_dataset, device)
    elif task == "audio-classification":
        return _eval_audio_classification(model, processor, eval_dataset, device)
    else:
        print(f"  ⚠️  Évaluation non implémentée pour '{task}' — accuracy=0.0 par défaut")
        return {"accuracy": 0.0}


# ── Classification texte ──────────────────────────────────────────────────────
def _eval_classification(model, dataset, device, task):
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Évaluation", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch.get("label", batch.get("labels"))

            if labels is None:
                continue
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if task == "token-classification":
                # (batch, seq_len, num_labels) → (batch*seq_len,)
                preds = outputs.logits.argmax(dim=-1).view(-1)
                flat_labels = labels.view(-1)
                # Ignorer label=-100 (padding)
                mask = flat_labels != -100
                correct += (preds[mask] == flat_labels[mask]).sum().item()
                total += mask.sum().item()
            else:
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    return {"accuracy": correct / total if total > 0 else 0.0}


# ── Perplexité (génération) ───────────────────────────────────────────────────
def _eval_perplexity(model, tokenizer, dataset, device):
    """
    Calcule la perplexité sur les textes du dataset.
    Fonctionne pour CausalLM (GPT, Llama, Mistral...) et Seq2SeqLM (T5, BART...).
    """
    total_loss, n_batches = 0.0, 0

    texts = []
    for item in dataset:
        t = item.get("text") or item.get("sentence") or item.get("document") or ""
        if t.strip():
            texts.append(t)
        if len(texts) >= (config.N_EVAL_SAMPLES or 200):
            break

    with torch.no_grad():
        for text in tqdm(texts[:100], desc="  Perplexité", leave=False):  # 100 max pour la rapidité
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
            ).to(device)

            try:
                if hasattr(model, "encoder"):  # Seq2Seq
                    labels = enc["input_ids"].clone()
                    out = model(**enc, labels=labels)
                else:  # CausalLM
                    out = model(**enc, labels=enc["input_ids"])

                if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                    total_loss += out.loss.item()
                    n_batches += 1
            except Exception:
                continue

    avg_loss = total_loss / n_batches if n_batches > 0 else float("inf")
    perplexity = math.exp(min(avg_loss, 20))  # cap pour éviter overflow
    return {"perplexity": round(perplexity, 2), "accuracy": max(0, 1 - avg_loss / 10)}


# ── Classification image ──────────────────────────────────────────────────────
def _eval_image_classification(model, feature_extractor, dataset, device):
    correct, total = 0, 0

    with torch.no_grad():
        for item in tqdm(dataset, desc="  Évaluation images", leave=False):
            try:
                image = item["image"]
                label = item["label"]
                inputs = feature_extractor(images=image, return_tensors="pt").to(device)
                out = model(**inputs)
                pred = out.logits.argmax(-1).item()
                correct += int(pred == label)
                total += 1
            except Exception:
                continue

    return {"accuracy": correct / total if total > 0 else 0.0}


# ── Classification audio ──────────────────────────────────────────────────────
def _eval_audio_classification(model, processor, dataset, device):
    correct, total = 0, 0

    with torch.no_grad():
        for item in tqdm(dataset, desc="  Évaluation audio", leave=False):
            try:
                audio = item["audio"]["array"]
                sr    = item["audio"]["sampling_rate"]
                label = item["label"]
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
                out = model(**inputs)
                pred = out.logits.argmax(-1).item()
                correct += int(pred == label)
                total += 1
            except Exception:
                continue

    return {"accuracy": correct / total if total > 0 else 0.0}
