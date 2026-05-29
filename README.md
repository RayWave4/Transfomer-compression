# Transformer Compression

---

## Structure

```
transformer_compression/
├── config.py          ←  FICHIER À MODIFIER
│
├── baseline.py        ←  Script indépendant — modèle original
├── quantization.py    ←  Script indépendant — INT8
├── pruning.py         ←  Script indépendant — L1 sparsity
├── distillation.py    ←  Script indépendant — teacher → student
│
├── run_all.py         ←  Optionnel — lance tout en séquence
├── plot_results.py    ←  Graphiques matplotlib
│
├── model_loader.py    ← Chargement universel HF (partagé)
├── evaluator.py       ← Évaluation adaptée par tâche (partagé)
├── utils.py           ← Taille, timing, tableaux (partagé)
└── requirements.txt
```

---

## Installation

```bash
# Créer un environnement virtuel
## Pour windows
python -m venv venv

source venv/bin/activate   # Linux/Mac | venv\Scripts\activate Windows
pip install -r requirements.txt
```

---

## Utilisation

### 1. Configurer le modèle (`config.py`)

```python
HF_TOKEN   = None     # ajouter le token pour modèles restreints
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
```

### 2. Lancer les scripts — chacun est 100% autonome

```bash
# Lancer chaque script de manière autonome :
python baseline.py
python quantization.py
python pruning.py
python distillation.py

# Ou tout en une fois (optionnel) :
python run_all.py

# Avec options :

python run_all.py --skip distillation # Lancer les script sans faire la distillation

python run_all.py --only quantization pruning # Lancer les script sans faire la quantization
```

### 3. Générer les graphiques

```bash
python plot_results.py
```

---

## Comportement intelligent des scripts

Comportement pour chaque script (`quantization.py`, `pruning.py`, `distillation.py`) :

| Situation | Comportement |
|---|---|
| `saves/baseline/` existe |  Réutilise le modèle local (rapide) |
| `saves/baseline/` absent |  Recharge depuis HuggingFace |
| `results/baseline.json` existe |  Affiche la comparaison baseline vs. compressé |
| `results/baseline.json` absent |  Affiche les métriques seules |

---

## Fichiers générés

```
results/
├── baseline.json        ← métriques baseline
├── quantization.json    ← métriques + comparaison baseline
├── pruning.json         ← métriques + comparaison baseline
├── distillation.json    ← métriques + comparaison baseline
├── all_results.json     ← récapitulatif global (run_all.py)
├── comparison_charts.png
└── accuracy_vs_size.png

saves/
├── baseline/            ← modèle original sauvegardé
├── quantization/        ← modèle quantizé
├── pruning/             ← modèle prunné
└── distillation/        ← student model
```

---

## Modèles compatibles

| Type | Exemples | Token requis |
|---|---|---|
| Classification texte | `distilbert-*`, `bert-*`, `roberta-*` | Non |
| Génération causale | `gpt2`, `meta-llama/Llama-3.2-1B` | Llama: oui |
| Génération seq2seq | `facebook/bart-base`, `t5-small` | Non |
| Classification image | `google/vit-base-patch16-224` | Non |
| Classification audio | `openai/whisper-tiny` | Non |

---

## Paramètres (`config.py`)

| Paramètre | Défaut | Description |
|---|---|---|
| `N_EVAL_SAMPLES` | 500 | Exemples d'évaluation |
| `PRUNING_SPARSITY` | 0.30 | Taux de pruning (0–1) |
| `DISTIL_TRAIN_SAMPLES` | 5000 | Exemples entraînement distillation |
| `DISTIL_EPOCHS` | 3 | Epochs distillation |
| `DISTIL_TEMPERATURE` | 4.0 | Température softmax |
| `DISTIL_ALPHA` | 0.5 | Poids CE vs KL |    
