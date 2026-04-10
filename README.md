# Skin Lesion Classification — DenseNet-201 🩺

A production-ready deep learning system that identifies **14 skin conditions** from dermoscopy images, backed by a FastAPI inference server and Test-Time Augmentation.

---

## ✦ What This Project Does

Skin disease diagnosis from images is difficult even for trained clinicians — especially when conditions look visually similar and labeled data is imbalanced. This pipeline tackles that problem end-to-end: from training a fine-tuned DenseNet-201 on a 14-class HuggingFace dataset, to serving predictions through a REST API with Swagger docs.

**Why it's non-trivial:**
- Rare classes (e.g. Dermatofibroma, Vascular lesions) are severely underrepresented
- 14-class classification with visually overlapping conditions demands more than a vanilla fine-tune
- Real clinical use requires not just accuracy, but calibrated confidence scores

---

## 🏆 Results at a Glance

| Mode | Accuracy | Weighted F1 | AUC (OvR) |
|---|---|---|---|
| Standard inference | 85.82% | — | — |
| With TTA (4 views) | — | **86.08%** | **98.06%** |

---

## 🗂 Dataset

**Source →** [`ahmed-ai/skin-lesions-classification-dataset`](https://huggingface.co/datasets/ahmed-ai/skin-lesions-classification-dataset) on HuggingFace

Official train / validation / test splits used without modification. Images are pre-cached as `.npy` arrays at 224×224×3 (uint8) to eliminate per-epoch decode overhead.

**14 target classes**

```
Actinic keratoses          Basal cell carcinoma       Benign keratosis-like lesions
Chickenpox                 Cowpox                     Dermatofibroma
HFMD                       Healthy                    Measles
Melanocytic nevi            Melanoma                   Monkeypox
Squamous cell carcinoma    Vascular lesions
```

---

## 🔬 Model Design

**Backbone →** DenseNet-201 pretrained on ImageNet (~20M parameters, 1920-dim feature output)

**Custom head**

```
backbone.forward_features(x)   →   [B, 1920, 7, 7]
AdaptiveAvgPool2d(1)            →   [B, 1920]
BatchNorm1d  →  Dropout(0.4)
Linear(1920 → 512)  →  ReLU
BatchNorm1d  →  Dropout(0.2)
Linear(512 → 14)                →   logits
```

---

## ⚙️ Training Pipeline

Training runs in two distinct phases to protect pretrained features early on, then gradually unlock the full network.

**Phase 1 — Head warm-up**
> Backbone frozen · 15 epochs · LR 5e-4 · OneCycleLR · no augmentation mixing

**Phase 2 — Full fine-tune**
> Backbone unfrozen · up to 45 epochs · differential LRs · early stopping

```python
# Differential learning rates in Phase 2
optimizer = AdamW([
    { "params": model.backbone.parameters(), "lr": 1e-5 },
    { "params": model.head.parameters(),     "lr": 1e-4 },
])

# Scheduler: 4-epoch linear warm-up → CosineAnnealingWarmRestarts
scheduler = SequentialLR([LinearWarmup(epochs=4), CosineAnnealingWarmRestarts(T0=8)])
```

**Techniques that make the difference**

| Technique | Role |
|---|---|
| `WeightedRandomSampler` | Balances class frequency at the batch level |
| Focal Loss (γ=2.0) + Label Smoothing (ε=0.1) | Focuses training on hard, misclassified examples |
| Mixup + CutMix | Synthesizes new training examples; CutMix preserves local texture for rare classes |
| AMP (GradScaler) | Mixed-precision training for faster GPU throughput |
| TTA — 4 views | Original · H-flip · V-flip · 90° rotation averaged at inference |
| Grad-CAM++ | Saliency maps on the final dense block for interpretability |

---

## 🛠 Getting Started

**1 — Clone**
```bash
git clone https://github.com/sherkhan05/skin-lesion-classification.git
cd skin-lesion-classification
```

**2 — Install**
```bash
pip install -r requirements.txt
```

**3 — Add the checkpoint**
```bash
# Default location (same folder)
cp /your/path/best_model_phase2.pth .

# Or override via env var
export MODEL_CHECKPOINT=/your/path/best_model_phase2.pth
```

---

## 🌐 Serving the API

```bash
uvicorn skin_lesion_api:app --host 127.0.0.1 --port 8000   # production
uvicorn skin_lesion_api:app --reload --port 8000          # dev / hot-reload
```

**Available routes**

```
GET   /health          →  model status, device, and class list
POST  /predict         →  all 14 class confidence scores
POST  /predict/top3    →  top 3 predictions only
GET   /docs            →  Swagger UI
```

---


> **Disclaimer —** For research and educational use only. Not a certified medical device. Always seek professional dermatological advice for clinical decisions.
