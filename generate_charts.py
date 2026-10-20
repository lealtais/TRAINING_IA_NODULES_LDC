import json
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ARTIFACTS_DIR = Path(r"C:\Users\taisl\.gemini\antigravity\brain\1226488b-1dc0-48d4-acd4-82171b09b897")
RESULTS_DIR = Path(r"c:\Users\taisl\Downloads\tcc\results")
MODEL_PATH = r"C:\Users\taisl\Downloads\best.pt"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Plot Training Metrics from Checkpoint
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
results = ckpt.get("train_results", {})
epochs = results.get("epoch", [])

if epochs:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, results.get("train/box_loss", []), label="Train Box Loss", color="blue")
    axes[0, 0].plot(epochs, results.get("val/box_loss", []), label="Val Box Loss", color="orange", linestyle="--")
    axes[0, 0].set_title("Box Loss (Localização)")
    axes[0, 0].set_xlabel("Épocas")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Classification Loss
    axes[0, 1].plot(epochs, results.get("train/cls_loss", []), label="Train Cls Loss", color="green")
    axes[0, 1].plot(epochs, results.get("val/cls_loss", []), label="Val Cls Loss", color="red", linestyle="--")
    axes[0, 1].set_title("Classification Loss")
    axes[0, 1].set_xlabel("Épocas")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP Metrics
    axes[1, 0].plot(epochs, results.get("metrics/mAP50(B)", []), label="mAP@50", color="purple", linewidth=2)
    axes[1, 0].plot(epochs, results.get("metrics/mAP50-95(B)", []), label="mAP@50-95", color="magenta", linestyle="--")
    axes[1, 0].set_title("Desempenho mAP (Mean Average Precision)")
    axes[1, 0].set_xlabel("Épocas")
    axes[1, 0].set_ylabel("mAP")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(epochs, results.get("metrics/precision(B)", []), label="Precisão", color="teal", linewidth=2)
    axes[1, 1].plot(epochs, results.get("metrics/recall(B)", []), label="Recall (Revocação)", color="coral", linewidth=2)
    axes[1, 1].set_title("Precisão vs Recall")
    axes[1, 1].set_xlabel("Épocas")
    axes[1, 1].set_ylabel("Taxa")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(ARTIFACTS_DIR / "training_curves.png"), dpi=150)
    plt.close()
    print("Saved training_curves.png")

# 2. Create Sample Predictions Montage (conf=0.25)
conf25_dir = RESULTS_DIR / "conf_25"
categories = [
    ("adenocarcinoma", "Adenocarcinoma (Câncer)"),
    ("large.cell.carcinoma", "Carcinoma de Grandes Células"),
    ("squamous.cell.carcinoma", "Carcinoma de Células Escamosas"),
    ("normal", "Pulmão Normal / Saudável")
]

fig, axes = plt.subplots(4, 3, figsize=(15, 18))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

for r, (cat_key, cat_title) in enumerate(categories):
    matching_files = sorted(list(conf25_dir.glob(f"{cat_key}_*.png")))
    for c in range(3):
        ax = axes[r, c]
        if c < len(matching_files):
            img = cv2.imread(str(matching_files[c]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"{cat_title}\nExemplo #{c+1}", fontsize=10, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "test_inference_samples.png"), dpi=150)
plt.close()
print("Saved test_inference_samples.png")

print("All artifact figures generated successfully!")
