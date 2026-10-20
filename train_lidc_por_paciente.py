# -*- coding: utf-8 -*-
"""
Treinamento YOLOv8s no LIDC-IDRI com Divisão por PACIENTE (Patient-Level Split)
===============================================================================
Este script corrige o vazamento de dados (data leakage) da divisão por fatia.
Aqui, sorteamos os PACIENTES entre Treino (85%) e Validação (15%).
Assim, todas as fatias de um mesmo paciente ficam exclusivamente no treino
ou exclusivamente na validação.
"""

import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# =============================================================================
# 1. CONFIGURAÇÕES DE CAMINHOS E HIPERPARÂMETROS
# =============================================================================
# Se estiver rodando no Colab/Kaggle, baixa automático via kagglehub se necessário
USAR_KAGGLEHUB = True

if USAR_KAGGLEHUB:
    import kagglehub
    caminho_download = Path(kagglehub.dataset_download("zhangweiled/lidcidri"))
    # Localiza a pasta LIDC-IDRI-slices dentro do download
    RAIZ = caminho_download / "LIDC-IDRI-slices"
    if not RAIZ.exists():
        RAIZ = Path("/root/.cache/kagglehub/datasets/zhangweiled/lidcidri/versions/1/LIDC-IDRI-slices")
else:
    # Caso já tenha a pasta baixada localmente no PC / PyCharm, coloque o caminho aqui:
    RAIZ = Path(r"C:\Users\taisl\Downloads\LIDC-IDRI-slices")

SAIDA = Path("./data_lidc_por_paciente/yolo")

# Passo 2 do planejamento: incluir fatias sem nódulo (background) para reduzir falsos positivos
INCLUIR_FATIAS_SEM_NODULO = True
PROPORCAO_VAL = 0.15
RANDOM_SEED = 42

# =============================================================================
# 2. PREPARAÇÃO DAS PASTAS NO FORMATO YOLO
# =============================================================================
if SAIDA.exists():
    shutil.rmtree(SAIDA)

for split in ("train", "val"):
    (SAIDA / "images" / split).mkdir(parents=True, exist_ok=True)
    (SAIDA / "labels" / split).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. DIVISÃO POR PACIENTE (EVITA DATA LEAKAGE ENTRE TREINO E VALIDAÇÃO)
# =============================================================================
pacientes = sorted([
    p for p in os.listdir(RAIZ)
    if (RAIZ / p).is_dir()
])

print(f"Total de pacientes encontrados: {len(pacientes)}")

pacientes_train, pacientes_val = train_test_split(
    pacientes,
    test_size=PROPORCAO_VAL,
    random_state=RANDOM_SEED
)

split_do_paciente = {}
for p in pacientes_train:
    split_do_paciente[p] = "train"
for p in pacientes_val:
    split_do_paciente[p] = "val"

print(f"Pacientes no TREINO:    {len(pacientes_train)} ({len(pacientes_train)/len(pacientes)*100:.1f}%)")
print(f"Pacientes na VALIDAÇÃO: {len(pacientes_val)} ({len(pacientes_val)/len(pacientes)*100:.1f}%)")

# Verificação de segurança: garante que nenhum paciente está nos dois conjuntos
intersecao = set(pacientes_train).intersection(set(pacientes_val))
assert len(intersecao) == 0, f"ERRO: Vazamento detectado nos pacientes: {intersecao}"
print("Verificação OK: 0 pacientes compartilhados entre treino e validação!\n")

# =============================================================================
# 4. PROCESSAMENTO DAS FATIAS E FUSÃO DAS 4 MÁSCARAS DOS RADIOLOGISTAS
# =============================================================================
stats = {
    "train": {"com_nodulo": 0, "sem_nodulo": 0},
    "val": {"com_nodulo": 0, "sem_nodulo": 0},
}

for paciente in pacientes:
    split = split_do_paciente[paciente]
    pasta_paciente = RAIZ / paciente

    for nodulo in sorted(os.listdir(pasta_paciente)):
        pasta_nodulo = pasta_paciente / nodulo
        pasta_imgs = pasta_nodulo / "images"
        if not pasta_imgs.exists():
            continue

        arquivos_img = sorted(os.listdir(pasta_imgs))

        for i, nome_img in enumerate(arquivos_img):
            # União lógica das 4 máscaras (Mask_1 U Mask_2 U Mask_3 U Mask_4)
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

            nome_saida = f"{paciente}_{nodulo}_{i:03d}.png"
            caminho_img_destino = SAIDA / "images" / split / nome_saida
            caminho_label_destino = SAIDA / "labels" / split / f"{Path(nome_saida).stem}.txt"

            # Caso a fatia não tenha marcação em nenhuma das 4 máscaras (fatia negativa/normal)
            if mascara_combinada is None or mascara_combinada.sum() == 0:
                if INCLUIR_FATIAS_SEM_NODULO:
                    shutil.copy2(pasta_imgs / nome_img, caminho_img_destino)
                    # No YOLO, imagem negativa (background) tem arquivo .txt vazio
                    caminho_label_destino.write_text("")
                    stats[split]["sem_nodulo"] += 1
                continue

            # Extrai a bounding box mínima da máscara combinada
            ys, xs = np.where(mascara_combinada)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            with Image.open(pasta_imgs / nome_img) as im:
                img_w, img_h = im.size

            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Evita caixas degeneradas (largura ou altura zero)
            if w <= 0 or h <= 0:
                continue

            shutil.copy2(pasta_imgs / nome_img, caminho_img_destino)
            caminho_label_destino.write_text(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            stats[split]["com_nodulo"] += 1

print("=== RESUMO DO DATASET GERADO (POR PACIENTE) ===")
for s in ("train", "val"):
    total_s = stats[s]["com_nodulo"] + stats[s]["sem_nodulo"]
    print(f"[{s.upper():5s}] Com nódulo: {stats[s]['com_nodulo']:5d} | Sem nódulo: {stats[s]['sem_nodulo']:4d} | Total: {total_s:5d}")

# =============================================================================
# 5. CRIAÇÃO DO ARQUIVO dataset.yaml
# =============================================================================
yaml_path = SAIDA / "dataset.yaml"
yaml_content = f"""path: {SAIDA.resolve()}
train: images/train
val: images/val
names:
  0: nodulo
"""
yaml_path.write_text(yaml_content, encoding="utf-8")
print(f"\nArquivo de configuração criado em: {yaml_path.resolve()}")

# =============================================================================
# 6. TREINAMENTO DO MODELO YOLOv8s
# =============================================================================
if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=50,
        imgsz=224,
        batch=32,
        patience=15,
        name="lidc_deteccao_por_paciente",
        optimizer="AdamW",
    )
    print("\nTreinamento concluído! Pesos salvos na pasta runs/detect/lidc_deteccao_por_paciente/weights/best.pt")
