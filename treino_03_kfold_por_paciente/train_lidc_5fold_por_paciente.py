# -*- coding: utf-8 -*-
"""
Treino 03 — YOLOv8s com Validação Cruzada 5-Fold por Paciente (LIDC-IDRI + Controles Saudáveis)
Inspirado na técnica de K-Fold + Votação (Ensemble) de Filipe Lauar (lealtais/projects).
Pronto para rodar no Google Colab (GPU).
"""

# No Google Colab, descomente a linha abaixo na 1ª célula:
# !pip install ultralytics kagglehub scikit-learn -q

import os
import shutil
import kagglehub
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

# =============================================================================
# 0) CONECTAR GOOGLE DRIVE LOGO NO INÍCIO (PARA SALVAR CADA FOLD ASSIM QUE ACABAR)
# =============================================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PASTA_DRIVE = Path("/content/drive/MyDrive/TCC_IA_NODULOS_LIDC/treino_03_5fold")
    PASTA_DRIVE.mkdir(parents=True, exist_ok=True)
    USAR_DRIVE = os.path.ismount("/content/drive")
    print(f"☁️ Google Drive conectado: {USAR_DRIVE} -> {PASTA_DRIVE}")
except ImportError:
    USAR_DRIVE = False
    PASTA_DRIVE = Path("./treino_03_5fold_backup")
    PASTA_DRIVE.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1) BAIXAR OS 3 BANCOS DE DADOS ONLINE (NO SERVIDOR DO COLAB)
# =============================================================================
path_lidc = kagglehub.dataset_download("zhangweiled/lidcidri")
RAIZ_LIDC = Path(path_lidc) / "LIDC-IDRI-slices"
if not RAIZ_LIDC.exists():
    RAIZ_LIDC = Path("/root/.cache/kagglehub/datasets/zhangweiled/lidcidri/versions/1/LIDC-IDRI-slices")

path_normal_1 = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")
path_normal_2 = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")

# =============================================================================
# 2) PRÉ-PROCESSAR TODAS AS IMAGENS UMA ÚNICA VEZ (LIDC + SAUDÁVEIS)
# =============================================================================
BASE_CACHE = Path("/content/cache_lidc_5fold")
if BASE_CACHE.exists():
    shutil.rmtree(BASE_CACHE)
(BASE_CACHE / "images").mkdir(parents=True, exist_ok=True)
(BASE_CACHE / "labels").mkdir(parents=True, exist_ok=True)

pacientes = sorted([p for p in os.listdir(RAIZ_LIDC) if (RAIZ_LIDC / p).is_dir()])
itens_por_paciente = {p: [] for p in pacientes}
n_ok, n_vazia = 0, 0

for paciente in pacientes:
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
                continue

            ys, xs = np.where(mascara_combinada)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            with Image.open(pasta_imgs / nome_img) as im:
                img_w, img_h = im.size

            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            if w <= 0 or h <= 0:
                continue

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h

            nome_saida = f"{paciente}_{nodulo}_{i:03d}.png"
            shutil.copy2(pasta_imgs / nome_img, BASE_CACHE / "images" / nome_saida)
            (BASE_CACHE / "labels" / f"{Path(nome_saida).stem}.txt").write_text(
                f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            )
            itens_por_paciente[paciente].append(nome_saida)
            n_ok += 1

print(f"LIDC-IDRI pronto: {len(pacientes)} pacientes | {n_ok} fatias com nódulo (puladas: {n_vazia})")

# Processar imagens de pulmão saudável (escala de cinza + .txt vazio)
imagens_normais = []
for base_normal in (path_normal_1, path_normal_2):
    for raiz_dir, _, arquivos in os.walk(base_normal):
        if "normal" in Path(raiz_dir).name.lower():
            for arq in sorted(arquivos):
                if arq.lower().endswith((".png", ".jpg", ".jpeg")):
                    imagens_normais.append(Path(raiz_dir) / arq)

nomes_normais = []
for idx, caminho_img in enumerate(imagens_normais):
    nome_saida = f"normal_saudavel_{idx:04d}.png"
    with Image.open(caminho_img) as im:
        im.convert("L").save(BASE_CACHE / "images" / nome_saida)
    (BASE_CACHE / "labels" / f"normal_saudavel_{idx:04d}.txt").write_text("")
    nomes_normais.append(nome_saida)

print(f"Pulmões saudáveis prontos: {len(nomes_normais)} imagens")

# =============================================================================
# 3) VALIDAÇÃO CRUZADA 5-FOLD EM NÍVEL DE PACIENTE (ZERO DATA LEAKAGE)
# =============================================================================
N_FOLDS = 5
kf_pacientes = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
kf_normais = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

splits_pacientes = list(kf_pacientes.split(pacientes))
splits_normais = list(kf_normais.split(nomes_normais))

resultados_folds = []

# Dica: Troque "yolov8s.pt" por "yolov8n.pt" ou "yolov8m.pt" se quiser comparar arquiteturas!
MODELO_BASE = "yolov8s.pt"

for fold in range(N_FOLDS):
    print(f"\n=======================================================")
    print(f"🚀 INICIANDO FOLD {fold + 1} / {N_FOLDS} ({MODELO_BASE})")
    print(f"=======================================================")

    train_p_idx, val_p_idx = splits_pacientes[fold]
    train_n_idx, val_n_idx = splits_normais[fold]

    pac_train = [pacientes[i] for i in train_p_idx]
    pac_val = [pacientes[i] for i in val_p_idx]

    # Garantia matemática de zero vazamento entre pacientes
    assert set(pac_train).isdisjoint(set(pac_val)), "Erro: paciente repetido no treino e validação!"

    lista_train = [img for p in pac_train for img in itens_por_paciente[p]] + [nomes_normais[i] for i in train_n_idx]
    lista_val = [img for p in pac_val for img in itens_por_paciente[p]] + [nomes_normais[i] for i in val_n_idx]

    # Criar arquivos .txt de índice para o YOLO (muito mais rápido que copiar 16.000 fotos 5 vezes!)
    fold_dir = Path(f"/content/folds/fold_{fold + 1}")
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    train_txt.write_text("\n".join(str(BASE_CACHE / "images" / f) for f in lista_train))
    val_txt.write_text("\n".join(str(BASE_CACHE / "images" / f) for f in lista_val))

    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text(
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"names:\n"
        f"  0: nodulo\n"
    )

    print(f"Fold {fold + 1}: {len(pac_train)} pacientes treino ({len(lista_train)} imgs) | {len(pac_val)} pacientes val ({len(lista_val)} imgs)")

    model = YOLO(MODELO_BASE)
    metrics = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=224,
        batch=32,
        patience=15,
        name=f"lidc_5fold_f{fold + 1}",
        optimizer="AdamW"
    )

    res = metrics.results_dict
    p = res["metrics/precision(B)"]
    r = res["metrics/recall(B)"]
    map50 = res["metrics/mAP50(B)"]
    map5095 = res["metrics/mAP50-95(B)"]
    resultados_folds.append({"fold": fold + 1, "P": p, "R": r, "mAP50": map50, "mAP50-95": map5095})
    print(f"✅ Fold {fold + 1} concluído -> P: {p:.3f} | R: {r:.3f} | mAP50: {map50:.3f} | mAP50-95: {map5095:.3f}")

    # Salvar best.pt e results.csv deste Fold imediatamente no Google Drive
    pasta_run = Path(f"/content/runs/detect/lidc_5fold_f{fold + 1}")
    best_src = pasta_run / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, PASTA_DRIVE / f"best_fold{fold + 1}.pt")
        shutil.copy2(pasta_run / "results.csv", PASTA_DRIVE / f"results_fold{fold + 1}.csv")
        print(f"☁️ Backup salvo: {PASTA_DRIVE / f'best_fold{fold + 1}.pt'}")

# =============================================================================
# 4) RESUMO FINAL DOS 5 FOLDS (MÉDIA ± DESVIO PADRÃO PARA O TCC)
# =============================================================================
print("\n================ RESULTADO FINAL 5-FOLD (POR PACIENTE) ================")
for rf in resultados_folds:
    print(f"Fold {rf['fold']}: P={rf['P']:.4f} | R={rf['R']:.4f} | mAP50={rf['mAP50']:.4f} | mAP50-95={rf['mAP50-95']:.4f}")

for key in ("P", "R", "mAP50", "mAP50-95"):
    vals = [rf[key] for rf in resultados_folds]
    print(f"MÉDIA {key:<8}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%")


# =============================================================================
# 5) EXEMPLO DE VOTAÇÃO DOS 5 MODELOS (ENSEMBLE) PARA USAR NA API PYTHON
# =============================================================================
def prever_com_votacao_5_folds(caminho_imagem, pasta_pesos, conf=0.25, votos_minimos=3):
    """
    Passa a tomografia nos 5 modelos treinados (best_fold1.pt ... best_fold5.pt).
    Só confirma um nódulo se pelo menos `votos_minimos` (ex.: 3 de 5) concordarem!
    """
    modelos = [YOLO(str(Path(pasta_pesos) / f"best_fold{k}.pt")) for k in range(1, 6)]
    deteccoes_por_modelo = 0
    todas_confs = []

    for m in modelos:
        res = m.predict(source=str(caminho_imagem), conf=conf, imgsz=224, verbose=False)[0]
        if len(res.boxes) > 0:
            deteccoes_por_modelo += 1
            todas_confs.extend([float(b.conf[0].cpu().numpy()) for b in res.boxes])

    confirmado = deteccoes_por_modelo >= votos_minimos
    return {
        "nodulo_confirmado": confirmado,
        "votos": f"{deteccoes_por_modelo}/5 modelos",
        "confianca_media": round(float(np.mean(todas_confs)), 3) if todas_confs else 0.0
    }
