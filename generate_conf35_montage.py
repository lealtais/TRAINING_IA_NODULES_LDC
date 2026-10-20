import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

ARTIFACTS_DIR = Path(r"C:\Users\taisl\.gemini\antigravity\brain\1226488b-1dc0-48d4-acd4-82171b09b897")
DATA_DIR = Path(r"c:\Users\taisl\Downloads\tcc\test_data")
MODEL_PATH = r"C:\Users\taisl\Downloads\best.pt"

model = YOLO(MODEL_PATH)

categories = [
    ("adenocarcinoma", "Adenocarcinoma (Maligno)"),
    ("large.cell.carcinoma", "Carcinoma de Grandes Células (Maligno)"),
    ("squamous.cell.carcinoma", "Carcinoma Epidermoide (Maligno)"),
    ("normal", "Tomografia Pulmonar Normal (Sem Nódulo)")
]

fig, axes = plt.subplots(4, 3, figsize=(15, 18))
plt.subplots_adjust(hspace=0.35, wspace=0.1)

for r, (cat_key, cat_title) in enumerate(categories):
    cat_dir = DATA_DIR / cat_key
    imgs = sorted(list(cat_dir.glob("*.png")))[:3]
    for c, img_path in enumerate(imgs):
        ax = axes[r, c]
        preds = model.predict(source=str(img_path), conf=0.35, imgsz=224, verbose=False)
        res_plotted = preds[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        num_det = len(preds[0].boxes)
        status_text = f"Detecção: {num_det} nódulo(s)" if num_det > 0 else "Nenhum nódulo detectado"
        
        ax.imshow(res_rgb)
        ax.set_title(f"{cat_title}\nExemplo #{c+1} - {status_text}", fontsize=9, fontweight="bold")
        ax.axis("off")

plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "optimal_threshold_conf35.png"), dpi=150)
plt.close()
print("Saved optimal_threshold_conf35.png")
