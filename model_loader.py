"""
model_loader.py — Chargement universel de n'importe quel modèle HuggingFace
============================================================================
Corrections appliquées :
  - dataset imdb → stanfordnlp/imdb  (nouveau format namespace/name obligatoire)
  - trust_remote_code retiré de load_dataset (déprécié)
  - LLaMA 1/2/3 : pad_token, dtype, tokenizer lent/rapide
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
    AutoModelForAudioClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)
from datasets import load_dataset
from huggingface_hub import login, HfApi
import config


# ── Mapping tâche → (AutoModel class, dataset par défaut) ────────────────────
TASK_REGISTRY = {
    "text-classification": {
        "model_class": AutoModelForSequenceClassification,
        "dataset": {"path": "stanfordnlp/imdb", "name": None, "split": "test",   # ← CORRIGÉ
                    "text_col": "text", "label_col": "label"},
        "input_type": "text",
    },
    "token-classification": {
        "model_class": AutoModelForTokenClassification,
        "dataset": {"path": "eriktks/conll2003", "name": None, "split": "validation",
                    "text_col": "tokens", "label_col": "ner_tags"},
        "input_type": "text",
    },
    "question-answering": {
        "model_class": AutoModelForQuestionAnswering,
        "dataset": {"path": "rajpurkar/squad", "name": None, "split": "validation",
                    "text_col": "question", "label_col": "answers"},
        "input_type": "text",
    },
    "text-generation": {
        "model_class": AutoModelForCausalLM,
        "dataset": {"path": "Salesforce/wikitext", "name": "wikitext-2-raw-v1", "split": "test",
                    "text_col": "text", "label_col": None},
        "input_type": "text",
    },
    "text2text-generation": {
        "model_class": AutoModelForSeq2SeqLM,
        "dataset": {"path": "EdinburghNLP/xsum", "name": None, "split": "test",
                    "text_col": "document", "label_col": "summary"},
        "input_type": "text",
    },
    "image-classification": {
        "model_class": AutoModelForImageClassification,
        "dataset": {"path": "imagenet-1k", "name": None, "split": "validation",
                    "text_col": "image", "label_col": "label"},
        "input_type": "image",
    },
    "audio-classification": {
        "model_class": AutoModelForAudioClassification,
        "dataset": {"path": "speech_commands", "name": "v0.01", "split": "validation",
                    "text_col": "audio", "label_col": "label"},
        "input_type": "audio",
    },
}


def authenticate():
    if config.HF_TOKEN:
        print("  🔐 Authentification HuggingFace...")
        login(token=config.HF_TOKEN)
        print("  ✓ Connecté.\n")
    else:
        print("  ℹ️  Pas de token HF — mode public uniquement.\n")


def detect_task(model_name: str) -> str:
    token = config.HF_TOKEN or True
    try:
        cfg = AutoConfig.from_pretrained(model_name, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger la config de '{model_name}'.\n"
            f"  → Vérifiez le nom du modèle et votre HF_TOKEN.\n"
            f"  Erreur : {e}"
        )

    arch = cfg.architectures[0].lower() if cfg.architectures else ""

    if any(k in arch for k in ["forcausalllm", "causalllm", "gpt", "llama", "mistral",
                                "gemma", "phi", "falcon", "mpt", "bloom"]):
        return "text-generation"
    if any(k in arch for k in ["forseq2seqlm", "bart", "t5", "pegasus", "marian"]):
        return "text2text-generation"
    if any(k in arch for k in ["forsequenceclassification", "classification"]):
        if hasattr(cfg, "image_size") or "vit" in arch or "resnet" in arch or "swin" in arch:
            return "image-classification"
        return "text-classification"
    if any(k in arch for k in ["fortokenclassification"]):
        return "token-classification"
    if any(k in arch for k in ["forquestionanswering"]):
        return "question-answering"
    if any(k in arch for k in ["foraudioclassification", "wav2vec", "whisper", "hubert"]):
        return "audio-classification"

    try:
        api = HfApi()
        info = api.model_info(model_name, token=config.HF_TOKEN)
        if info.pipeline_tag and info.pipeline_tag in TASK_REGISTRY:
            return info.pipeline_tag
    except Exception:
        pass

    print("  ⚠️  Tâche non détectée automatiquement — défaut: text-classification")
    return "text-classification"


# ── LLaMA helpers ─────────────────────────────────────────────────────────────
def _detect_llama_version(model_name: str) -> int:
    name_lower = model_name.lower()
    if "llama-3" in name_lower or "llama3" in name_lower or "meta-llama-3" in name_lower:
        return 3
    if "llama-2" in name_lower or "llama2" in name_lower or "meta-llama-2" in name_lower:
        return 2
    if "llama-1" in name_lower or "huggyllama" in name_lower or "decapoda" in name_lower:
        return 1
    try:
        cfg = AutoConfig.from_pretrained(model_name, token=config.HF_TOKEN or True)
        if getattr(cfg, "vocab_size", 32000) > 100000:
            return 3
    except Exception:
        pass
    return 2


def _is_llama_family(model_name: str) -> bool:
    return any(k in model_name.lower() for k in ["llama", "huggyllama", "decapoda"])


def load_processor(model_name: str, task: str):
    token = config.HF_TOKEN or True
    kwargs = {"token": token}
    input_type = TASK_REGISTRY[task]["input_type"]

    if input_type == "image":
        try:
            return AutoFeatureExtractor.from_pretrained(model_name, **kwargs)
        except Exception:
            return AutoProcessor.from_pretrained(model_name, **kwargs)

    elif input_type == "audio":
        try:
            return AutoProcessor.from_pretrained(model_name, **kwargs)
        except Exception:
            return AutoFeatureExtractor.from_pretrained(model_name, **kwargs)

    else:
        llama_version = _detect_llama_version(model_name) if _is_llama_family(model_name) else None
        tok_kwargs = dict(kwargs)

        if llama_version == 1:
            tok_kwargs["use_fast"] = False
            print("  ℹ️  LLaMA 1 détecté — tokenizer lent (SentencePiece)")
        if llama_version == 3:
            print("  ℹ️  LLaMA 3 détecté — tokenizer tiktoken (vocab 128k)")

        tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
            print(f"  ℹ️  pad_token non défini → eos_token utilisé ('{tok.eos_token}')")

        if task in ("text-generation", "text2text-generation"):
            tok.padding_side = "left"

        return tok


def _resolve_torch_dtype(model_name: str, task: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if task not in ("text-generation", "text2text-generation"):
        return torch.float32
    if _is_llama_family(model_name):
        v = _detect_llama_version(model_name)
        if v == 3:
            if torch.cuda.is_bf16_supported():
                print("  ℹ️  LLaMA 3 → dtype bfloat16")
                return torch.bfloat16
            else:
                print("  ⚠️  bfloat16 non supporté → float16")
                return torch.float16
        return torch.float16
    return torch.float16


def load_model(model_name: str, task: str):
    token = config.HF_TOKEN or True
    model_class = TASK_REGISTRY[task]["model_class"]
    kwargs = {"token": token}

    if task in ("text-generation", "text2text-generation"):
        kwargs["torch_dtype"] = _resolve_torch_dtype(model_name, task)
        kwargs["device_map"] = "auto"
        kwargs["trust_remote_code"] = True

    model = model_class.from_pretrained(model_name, **kwargs)
    model.eval()

    if _is_llama_family(model_name):
        v = _detect_llama_version(model_name)
        print(f"  ✓ LLaMA v{v} — vocab={model.config.vocab_size}, "
              f"layers={model.config.num_hidden_layers}, "
              f"heads={model.config.num_attention_heads}")

    return model


def load_eval_dataset(task: str, processor):
    ds_cfg = config.DATASET_CONFIG or TASK_REGISTRY[task]["dataset"]
    input_type = TASK_REGISTRY[task]["input_type"]

    print(f"  Dataset: {ds_cfg['path']} ({ds_cfg.get('name') or 'default'}) — split: {ds_cfg['split']}")

    try:
        raw = load_dataset(
            ds_cfg["path"],
            ds_cfg.get("name"),
            split=ds_cfg["split"],
            token=config.HF_TOKEN or True,
            # trust_remote_code retiré — déprécié dans les versions récentes
        )
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le dataset '{ds_cfg['path']}'.\n  Erreur : {e}")

    if config.N_EVAL_SAMPLES:
        raw = raw.select(range(min(config.N_EVAL_SAMPLES, len(raw))))

    if input_type == "text":
        text_col  = ds_cfg["text_col"]
        label_col = ds_cfg.get("label_col")

        # Pour les tâches génératives, l'évaluateur tokenise lui-même — retourner les textes bruts
        if task in ("text-generation", "text2text-generation"):
            return raw

        def tokenize(batch):
            return processor(
                batch[text_col],
                truncation=True,
                padding="max_length",
                max_length=config.MAX_LENGTH,
            )

        dataset = raw.map(tokenize, batched=True, remove_columns=[
            c for c in raw.column_names if c not in [label_col, text_col]
        ])

        format_cols = [c for c in ["input_ids", "attention_mask", label_col]
                       if c and c in dataset.column_names]
        dataset.set_format(type="torch", columns=format_cols)

        if label_col and label_col != "label" and label_col in dataset.column_names:
            dataset = dataset.rename_column(label_col, "label")

        return dataset

    return raw


def load_all():
    authenticate()

    print(f"  Modèle : {config.MODEL_NAME}")
    task = detect_task(config.MODEL_NAME)
    print(f"  Tâche détectée : {task}")

    print("  Chargement du processor...")
    processor = load_processor(config.MODEL_NAME, task)

    print("  Chargement du modèle...")
    model = load_model(config.MODEL_NAME, task)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Paramètres : {param_count:.1f}M")

    print("  Chargement du dataset...")
    eval_dataset = load_eval_dataset(task, processor)
    print(f"  Exemples d'évaluation : {len(eval_dataset)}\n")

    return model, processor, eval_dataset, task