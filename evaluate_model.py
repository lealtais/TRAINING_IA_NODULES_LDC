import os
import requests
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(r"c:\Users\taisl\Downloads\tcc")
DATA_DIR = BASE_DIR / "test_data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = r"C:\Users\taisl\Downloads\best.pt"

print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")
print(f"Classes: {model.names}")
print(f"Model task: {model.task}")

# GitHub repo with Chest CT-Scan images
REPO = "hallowshaw/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning"
BRANCH = "main"
HEADERS = {"User-Agent": "Mozilla/5.0"}

tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
print("Fetching dataset structure from GitHub...")
resp = requests.get(tree_url, headers=HEADERS)
tree_data = resp.json().get("tree", [])

categories = ["adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma", "normal"]
samples_per_cat = 10 # download 10 sample images per category

downloaded_files = {}

for cat in categories:
    cat_dir = DATA_DIR / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    cat_files = [f["path"] for f in tree_data if f["path"].startswith(f"dataset/test/{cat}/") and f["path"].lower().endswith((".png", ".jpg", ".jpeg"))]
    
    print(f"Found {len(cat_files)} images for category '{cat}'")
    selected_files = cat_files[:samples_per_cat]
    downloaded_files[cat] = []
    
    for i, file_path in enumerate(selected_files):
        filename = Path(file_path).name
        clean_name = filename.replace(" ", "_").replace("(", "").replace(")", "")
        local_path = cat_dir / clean_name
        
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{file_path}"
        r = requests.get(raw_url, headers=HEADERS)
        if r.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(r.content)
            downloaded_files[cat].append(local_path)
            print(f"  [OK] Downloaded {cat}/{clean_name}")
        else:
            print(f"  [FAIL] Failed to download {raw_url} (status: {r.status_code})")

print("\n--- Running Model Inference on Downloaded Images ---")

stats = {}

for conf_thresh in [0.25, 0.15, 0.10, 0.05]:
    conf_dir = RESULTS_DIR / f"conf_{int(conf_thresh*100)}"
    conf_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n================ CONFIDENCE THRESHOLD = {conf_thresh} ================")
    stats[str(conf_thresh)] = {}
    
    for cat, file_paths in downloaded_files.items():
        stats[str(conf_thresh)][cat] = {
            "total": len(file_paths),
            "detected": 0,
            "total_boxes": 0,
            "confidences": [],
            "details": []
        }
        
        for p in file_paths:
            results = model.predict(source=str(p), conf=conf_thresh, imgsz=224, verbose=False)
            res = results[0]
            boxes = res.boxes
            num_boxes = len(boxes)
            
            box_info = []
            if num_boxes > 0:
                stats[str(conf_thresh)][cat]["detected"] += 1
                stats[str(conf_thresh)][cat]["total_boxes"] += num_boxes
                for b in boxes:
                    conf = float(b.conf[0].cpu().numpy())
                    cls_id = int(b.cls[0].cpu().numpy())
                    cls_name = model.names.get(cls_id, str(cls_id))
                    xyxy = [round(float(x), 2) for x in b.xyxy[0].cpu().numpy()]
                    stats[str(conf_thresh)][cat]["confidences"].append(conf)
                    box_info.append({"class": cls_name, "conf": round(conf, 4), "box": xyxy})
            
            stats[str(conf_thresh)][cat]["details"].append({
                "file": p.name,
                "num_boxes": num_boxes,
                "boxes": box_info
            })
            
            # Save plotted image
            res_plotted = res.plot()
            out_img_path = conf_dir / f"{cat}_{p.name}"
            cv2.imwrite(str(out_img_path), res_plotted)
            
        det_rate = (stats[str(conf_thresh)][cat]["detected"] / max(1, stats[str(conf_thresh)][cat]["total"])) * 100
        avg_conf = float(np.mean(stats[str(conf_thresh)][cat]["confidences"])) if stats[str(conf_thresh)][cat]["confidences"] else 0.0
        print(f"Category: {cat:25s} | Detected: {stats[str(conf_thresh)][cat]['detected']}/{stats[str(conf_thresh)][cat]['total']} ({det_rate:.1f}%) | Avg Conf: {avg_conf:.3f} | Total Boxes: {stats[str(conf_thresh)][cat]['total_boxes']}")

# Save summary JSON
with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"\nAll results and annotated images saved in {RESULTS_DIR}")
