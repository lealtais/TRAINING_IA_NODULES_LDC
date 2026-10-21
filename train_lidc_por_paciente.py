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

        # Converte para escala de cinza (L) igual ao LIDC-IDRI e salva como PNG
        with Image.open(caminho_img) as im:
            im.convert("L").save(destino_img)

        # Arquivo .txt em branco indica ao YOLO que não há nódulo na imagem
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
