from pathlib import Path
from PIL import Image
from ultralytics import YOLO

m = YOLO(r"C:\Users\China Link\Downloads\best.pt")
d = Path(r"C:\Users\China Link\.gemini\antigravity\brain\1191f4fd-54de-4f8a-8588-f62bbf42b5f3\scratch\test_data_40")
REPO_ASSETS = Path(r"C:\Users\China Link\.gemini\antigravity\scratch\TRAINING_IA_NODULES_LDC\assets")

def predict_with_patches(im_path, conf):
    im = Image.open(im_path).convert("L")
    w, h = im.size
    confs = []
    for sy in [0.15, 0.30, 0.45]:
        for sx in [0.15, 0.35, 0.55]:
            crop = im.crop((int(w * sx), int(h * sy), int(w * (sx + 0.35)), int(h * (sy + 0.35))))
            r = m.predict(crop, conf=conf, imgsz=224, verbose=False)[0]
            for b in r.boxes:
                confs.append(round(float(b.conf[0].cpu().numpy()), 3))
    return confs

print("=== TABELA 40 IMAGENS CLINICAS (COM JANELA DE ZOOM ESCALA LIDC-IDRI) ===")
for conf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    res_cat = {}
    for cat in ["adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma", "normal"]:
        det = 0
        for p in sorted(list((d / cat).glob("*")))[:10]:
            if len(predict_with_patches(p, conf)) > 0:
                det += 1
        res_cat[cat] = det
    a = res_cat["adenocarcinoma"]
    l = res_cat["large.cell.carcinoma"]
    s = res_cat["squamous.cell.carcinoma"]
    n = res_cat["normal"]
    print(f"conf={conf:.2f} | Adeno: {a}/10 | Large: {l}/10 | Squamous: {s}/10 | Total Cancer: {a+l+s}/30 | Normal (FP): {n}/10 ({n*10}%)")

print("\n=== 4 IMAGENS EXTERNAS (COM JANELA DE ZOOM ESCALA LIDC-IDRI, conf=0.25) ===")
for name in ["teste_01_original.jpg", "teste_02_original.jpg", "teste_03_original.png", "teste_04_original.png"]:
    c = predict_with_patches(REPO_ASSETS / name, 0.25)
    print(f"{name}: {len(c)} deteccoes -> max={max(c) if c else 0.0} | todas={c}")
