import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "HFMD",
    "Healthy",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Inference transform (no augmentation) ─────────────────────────────────────
inference_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# TTA transforms
TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.VerticalFlip(p=1),   A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.RandomRotate90(p=1), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
]


# ── Model definition (must match training exactly) ────────────────────────────
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet201", pretrained=False, num_classes=0
        )
        in_feat = self.backbone.num_features   # 1920

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(x))


# ── Loader ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str = "cpu") -> SkinLesionModel:
    model = SkinLesionModel(num_classes=NUM_CLASSES, dropout=0.4)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoint
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: SkinLesionModel,
            pil_image: Image.Image,
            device: str = "cpu",
            use_tta: bool = True) -> dict:
    """
    Run inference on a single PIL image.
    Returns dict with predicted class, confidence, and all class probabilities.
    """
    img_np = np.array(pil_image.convert("RGB"))

    if use_tta:
        probs = torch.zeros(NUM_CLASSES, device=device)
        for tfm in TTA_TRANSFORMS:
            tensor = tfm(image=img_np)["image"].unsqueeze(0).to(device)
            probs += F.softmax(model(tensor).squeeze(0), dim=0)
        probs /= len(TTA_TRANSFORMS)
    else:
        tensor = inference_transform(image=img_np)["image"].unsqueeze(0).to(device)
        probs  = F.softmax(model(tensor).squeeze(0), dim=0)

    probs_np  = probs.cpu().numpy()
    pred_idx  = int(probs_np.argmax())

    return {
        "predicted_class"   : CLASS_NAMES[pred_idx],
        "confidence"        : round(float(probs_np[pred_idx]) * 100, 2),
        "class_probabilities": {
            name: round(float(p) * 100, 4)
            for name, p in zip(CLASS_NAMES, probs_np)
        },
    }
