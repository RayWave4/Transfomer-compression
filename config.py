
#config.py — Configuration centrale du projet
#Modifie CE FICHIER uniquement pour changer de modèle ou de tâche.

# Authentification HuggingFace
HF_TOKEN = None  # ex: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Modèle cible 
MODEL_NAME = "distilbert/distilgpt2"
# distilbert-base-uncased-finetuned-sst-2-english

# Dataset d'évaluation
# None = détection automatique | Exemple manuel :
# {"path": "glue", "name": "sst2", "split": "validation", "text_col": "sentence", "label_col": "label"}
# Pour distilgpt2 (text-generation) → WikiText-2 utilisé automatiquement
DATASET_CONFIG = None

# Paramètres généraux
N_EVAL_SAMPLES = 100   # plafond réel dans evaluator.py (_eval_perplexity texts[:100])
BATCH_SIZE     = 16    # réduit pour éviter OOM sur CPU
MAX_LENGTH     = 512   # plus long = perplexité plus représentative pour un LM
SEED           = 42

# Pruning 
PRUNING_SPARSITY = 0.30   # 0.0 à 1.0

# Knowledge Distillation
DISTIL_TRAIN_SAMPLES  = 5000
DISTIL_EPOCHS         = 3
DISTIL_LR             = 5e-5
DISTIL_TEMPERATURE    = 4.0
DISTIL_ALPHA          = 0.5
DISTIL_STUDENT_LAYERS = 2
DISTIL_STUDENT_DIM    = 256
