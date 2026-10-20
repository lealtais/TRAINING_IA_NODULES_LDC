# -*- coding: utf-8 -*-
!pip install ultralytics kagglehub scikit-learn -q

import os
import shutil
import kagglehub
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# 1) Baixar o banco de dados online (no servidor do Colab)
path = kagglehub.dataset_download("zhangweiled/lidcidri")
print("Path to dataset files:", path)

RAIZ = Path(path) / "LIDC-IDRI-slices"
if not RAIZ.exists():
    RAIZ = Path("/root/.cache/kagglehub/datasets/zhangweiled/lidcidri/versions/1/LIDC-IDRI-slices")

SAIDA = Path("/content/data_lidc/yolo")

if SAIDA.exists():
    shutil.rmtree(SAIDA)

for split in ("train", "val"):
    (SAIDA / "images" / split).mkdir(parents=True, exist_ok=True)
    (SAIDA / "labels" / split).mkdir(parents=True, exist_ok=True)

# 2) Separar treino/validação POR PACIENTE (e não por fatia)
pacientes = sorted([p for p in os.listdir(RAIZ) if (RAIZ / p).is_dir()])
print("Total de pacientes:", len(pacientes))

pacientes_train, pacientes_val = train_test_split(pacientes, test_size=0.15, random_state=42)

split_de = {}
for p in pacientes_train:
    split_de[p] = "train"
for p in pacientes_val:
    split_de[p] = "val"

print(f"Pacientes no treino: {len(pacientes_train)} | Pacientes na validação: {len(pacientes_val)}")

# 3) Processar as fatias combinando as 4 máscaras (igual ao treino original)
n_ok, n_vazia = 0, 0

for paciente in pacientes:
    split = split_de[paciente]
    pasta_paciente = RAIZ / paciente

    for nodulo in sorted(os.listdir(pasta_paciente)):
        pasta_nodulo = pasta_paciente / nodulo
        pasta_imgs = pasta_nodulo / "images"
        if not pasta_imgs.exists():
            continue

        arquivos_img = sorted(os.listdir(pasta_imgs))

        for i, nome_img in enumerate(arquivos_img):
            # combina as 4 máscaras
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
                continue  # pula fatia sem máscara válida

            ys, xs = np.where(mascara_combinada)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            with Image.open(pasta_imgs / nome_img) as im:
                img_w, img_h = im.size

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            nome_saida = f"{paciente}_{nodulo}_{i:03d}.png"
            shutil.copy2(pasta_imgs / nome_img, SAIDA / "images" / split / nome_saida)

            label_path = SAIDA / "labels" / split / (Path(nome_saida).stem + ".txt")
            label_path.write_text(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            n_ok += 1

print("Fatias processadas (com nódulo):", n_ok)
print("Fatias puladas (máscara vazia):", n_vazia)

# 4) Criar dataset.yaml e treinar
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
    name="lidc_deteccao_paciente",
    optimizer="AdamW"
)
