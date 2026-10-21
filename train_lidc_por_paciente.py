# -*- coding: utf-8 -*-
!pip install ultralytics kagglehub scikit-learn requests opencv-python -q

import os
import json
import shutil
import requests
import kagglehub
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from google.colab import drive, files

# =============================================================================
# 0) CONECTAR AO GOOGLE DRIVE LOGO NO INÍCIO (GARANTE QUE NADA SE PERCA)
# =============================================================================
drive.mount("/content/drive", force_remount=True)
PASTA_DRIVE = Path("/content/drive/MyDrive/TCC_NODULOS_LIDC_PACIENTE")
PASTA_DRIVE.mkdir(parents=True, exist_ok=True)
print(f"✅ Google Drive conectado! Tudo será salvo automaticamente em: {PASTA_DRIVE}")

# =============================================================================
# 1) BAIXAR OS BANCOS DE DADOS ONLINE (NO SERVIDOR DO COLAB)
# =============================================================================
# Banco 1: LIDC-IDRI (fatias com nódulos anotados por 4 radiologistas)
path_lidc = kagglehub.dataset_download("zhangweiled/lidcidri")
RAIZ_LIDC = Path(path_lidc) / "LIDC-IDRI-slices"
if not RAIZ_LIDC.exists():
    RAIZ_LIDC = Path("/root/.cache/kagglehub/datasets/zhangweiled/lidcidri/versions/1/LIDC-IDRI-slices")

# Bancos 2 e 3: Tomografias de Pulmão Saudável / Normal (sem nódulo)
path_normal_1 = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")
path_normal_2 = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")

SAIDA = Path("/content/data_lidc/yolo")
if SAIDA.exists():
    shutil.rmtree(SAIDA)

for split in ("train", "val"):
    (SAIDA / "images" / split).mkdir(parents=True, exist_ok=True)
    (SAIDA / "labels" / split).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2) SEPARAR PACIENTES DO LIDC-IDRI EM TREINO (85%) E VALIDAÇÃO (15%)
# =============================================================================
pacientes = sorted([p for p in os.listdir(RAIZ_LIDC) if (RAIZ_LIDC / p).is_dir()])
pacientes_train, pacientes_val = train_test_split(pacientes, test_size=0.15, random_state=42)

split_de = {}
for p in pacientes_train: split_de[p] = "train"
for p in pacientes_val:   split_de[p] = "val"

print(f"Pacientes LIDC no treino: {len(pacientes_train)} | na validação: {len(pacientes_val)}")

# =============================================================================
# 3) PROCESSAR FATIAS COM NÓDULO (LIDC-IDRI)
# =============================================================================
n_ok, n_vazia = 0, 0

for paciente in pacientes:
    split = split_de[paciente]
    pasta_paciente = RAIZ_LIDC / paciente

    for nodulo in sorted(os.listdir(pasta_paciente)):
        pasta_nodulo = pasta_paciente / nodulo
        pasta_imgs = pasta_nodulo / "images"
        if not pasta_imgs.exists():
            continue

        arquivos_img = sorted(os.listdir(pasta_imgs))

        for i, nome_img in enumerate(arquivos_img):
            mascara_combinada = None
            for k in range(4):
                pasta_mask = pasta_nodulo / f"mask-{k}"
                if not pasta_mask.exists():
                    continue
                arquivos_mask = sorted(os.listdir(pasta_mask))
                if i >= len(arquivos_mask):
                    continue
                m = np.array(Image.open(pasta_mask / arquivos_mask[i]).convert("L")) > 127
                mascara_combinada = m if mascara_combinada is None else (mascara_combinada | m)

            if mascara_combinada is None or mascara_combinada.sum() == 0:
                n_vazia += 1
                continue  # pula as 62 fatias com erro de máscara vazia

            ys, xs = np.where(mascara_combinada)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            with Image.open(pasta_imgs / nome_img) as im:
                img_w, img_h = im.size

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            if w <= 0 or h <= 0:
                continue

            nome_saida = f"{paciente}_{nodulo}_{i:03d}.png"
            shutil.copy2(pasta_imgs / nome_img, SAIDA / "images" / split / nome_saida)

            label_path = SAIDA / "labels" / split / (Path(nome_saida).stem + ".txt")
            label_path.write_text(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            n_ok += 1

print(f"Fatias COM nódulo processadas: {n_ok} (puladas: {n_vazia})")

# =============================================================================
# 4) ADICIONAR IMAGENS DE PULMÃO SAUDÁVEL / SEM NÓDULO (.txt VAZIO)
# =============================================================================
imagens_normais = []
for base_normal in (path_normal_1, path_normal_2):
    for raiz_dir, _, arquivos in os.walk(base_normal):
        if "normal" in Path(raiz_dir).name.lower():
            for arq in sorted(arquivos):
                if arq.lower().endswith((".png", ".jpg", ".jpeg")):
                    imagens_normais.append(Path(raiz_dir) / arq)

norm_train, norm_val = train_test_split(imagens_normais, test_size=0.15, random_state=42)

for split, lista_imgs in (("train", norm_train), ("val", norm_val)):
    for idx, caminho_img in enumerate(lista_imgs):
        nome_saida = f"normal_saudavel_{idx:04d}.png"
        destino_img = SAIDA / "images" / split / nome_saida
        destino_lbl = SAIDA / "labels" / split / f"normal_saudavel_{idx:04d}.txt"

        with Image.open(caminho_img) as im:
            im.convert("L").save(destino_img)

        destino_lbl.write_text("")

print(f"Imagens de pulmão saudável adicionadas: {len(imagens_normais)} (Treino: {len(norm_train)} | Val: {len(norm_val)})")

# =============================================================================
# 5) CRIAR dataset.yaml E TREINAR O MODELO YOLOv8s
# =============================================================================
yaml_content = """path: /content/data_lidc/yolo
train: images/train
val: images/val
names:
  0: nodulo
"""

with open("/content/data_lidc/yolo/dataset.yaml", "w") as f:
    f.write(yaml_content)

model_lidc = YOLO("yolov8s.pt")
model_lidc.train(
    data="/content/data_lidc/yolo/dataset.yaml",
    epochs=50,
    imgsz=224,
    batch=32,
    patience=15,
    name="lidc_paciente_com_saudaveis",
    optimizer="AdamW"
)

PASTA_TREINO = Path("/content/runs/detect/lidc_paciente_com_saudaveis")
CAMINHO_BEST = PASTA_TREINO / "weights" / "best.pt"

# =============================================================================
# 6) BAIXAR FOTOS DE CÂNCER DE PULMÃO DA INTERNET E TESTAR O NOVO best.pt
# =============================================================================
print("\n=== 6) BAIXANDO IMAGENS DE CÂNCER DE PULMÃO DA INTERNET PARA TESTE ===")
PASTA_TESTE_NET = Path("/content/testes_internet")
PASTA_ORIGINAIS = PASTA_TESTE_NET / "originais"
PASTA_RESULTADOS = PASTA_TESTE_NET / "predicoes"
PASTA_ORIGINAIS.mkdir(parents=True, exist_ok=True)
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
REPO = "hallowshaw/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning"
BRANCH = "main"
tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
tree_data = requests.get(tree_url, headers=HEADERS, timeout=20).json().get("tree", [])

categorias = ["adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma", "normal"]
arquivos_teste = {}

for cat in categorias:
    cat_dir = PASTA_ORIGINAIS / cat.replace(".", "_")
    cat_dir.mkdir(parents=True, exist_ok=True)
    arquivos_teste[cat] = []

    cat_files = [
        f["path"] for f in tree_data
        if f["path"].startswith(f"dataset/test/{cat}/")
        and f["path"].lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for file_path in cat_files[:10]:
        clean_name = Path(file_path).name.replace(" ", "_").replace("(", "").replace(")", "")
        local_path = cat_dir / clean_name
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{file_path}"
        r = requests.get(raw_url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            local_path.write_bytes(r.content)
            arquivos_teste[cat].append(local_path)

modelo_final = YOLO(str(CAMINHO_BEST))
resumo_testes = {}

for limiar in [0.20, 0.25, 0.30, 0.35, 0.40]:
    conf_dir = PASTA_RESULTADOS / f"conf_{int(limiar * 100)}"
    conf_dir.mkdir(parents=True, exist_ok=True)
    resumo_testes[str(limiar)] = {}
    print(f"\n--- Resultados no Limiar conf = {limiar} ---")

    for cat, lista_imgs in arquivos_teste.items():
        detectados = 0
        confs = []
        for p in lista_imgs:
            res = modelo_final.predict(source=str(p), conf=limiar, imgsz=224, verbose=False)[0]
            if len(res.boxes) > 0:
                detectados += 1
                for b in res.boxes:
                    confs.append(float(b.conf[0].cpu().numpy()))
            cv2.imwrite(str(conf_dir / f"{cat}_{p.name}"), res.plot())

        taxa = (detectados / max(1, len(lista_imgs))) * 100
        media_c = float(np.mean(confs)) if confs else 0.0
        resumo_testes[str(limiar)][cat] = {
            "detectados": f"{detectados}/{len(lista_imgs)}",
            "taxa_pct": round(taxa, 1),
            "conf_media": round(media_c, 3)
        }
        print(f"  {cat:25s} -> Detectou: {detectados:2d}/{len(lista_imgs):2d} ({taxa:5.1f}%) | Conf. Média: {media_c:.3f}")

(PASTA_TESTE_NET / "resumo_testes_internet.json").write_text(
    json.dumps(resumo_testes, indent=2, ensure_ascii=False), encoding="utf-8"
)

# =============================================================================
# 7) SALVAR NO GOOGLE DRIVE E BAIXAR DIRETO PARA O SEU PC (PASTA DOWNLOADS)
# =============================================================================
shutil.copytree(PASTA_TREINO, PASTA_DRIVE / "lidc_paciente_com_saudaveis", dirs_exist_ok=True)
shutil.copytree(PASTA_TESTE_NET, PASTA_DRIVE / "testes_internet", dirs_exist_ok=True)
shutil.copy2(CAMINHO_BEST, PASTA_DRIVE / "best.pt")
print(f"\n✅ Tudo salvo no seu Google Drive em: {PASTA_DRIVE}")

# Compacta tudo (pesos best.pt + gráficos + fotos da internet testadas) e baixa pro PC!
PACOTE_FINAL = Path("/content/pacote_completo_tcc")
PACOTE_FINAL.mkdir(parents=True, exist_ok=True)
shutil.copy2(CAMINHO_BEST, PACOTE_FINAL / "best.pt")
shutil.copytree(PASTA_TREINO, PACOTE_FINAL / "treino_yolo", dirs_exist_ok=True)
shutil.copytree(PASTA_TESTE_NET, PACOTE_FINAL / "testes_internet", dirs_exist_ok=True)

shutil.make_archive("/content/RESULTADO_COMPLETO_TCC", "zip", PACOTE_FINAL)
files.download(str(CAMINHO_BEST))
files.download("/content/RESULTADO_COMPLETO_TCC.zip")
print("🎉 Downloads de 'best.pt' e 'RESULTADO_COMPLETO_TCC.zip' iniciados para o seu PC!")
