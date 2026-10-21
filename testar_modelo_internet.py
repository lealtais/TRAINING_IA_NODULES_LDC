# -*- coding: utf-8 -*-
"""
Script completo para:
1. Baixar imagens reais de Tomografia de Câncer de Pulmão (Adenocarcinoma,
   Carcinoma de Grandes Células, Carcinoma de Células Escamosas, Casos Extras da
   Internet/Wikimedia e Pulmões Saudáveis de controle).
2. Carregar o modelo `best.pt` salvo no PC (ex: na pasta Downloads) e testar
   em todas as imagens baixadas, salvando as fotos com as caixas desenhadas
   e um relatório completo de acertos e confiança.
"""

import os
import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "testes_internet" / "imagens_originais"
RESULTS_DIR = BASE_DIR / "testes_internet" / "resultados_deteccao"

TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def baixar_imagens_internet(qtd_por_categoria=10):
    print("=== 1. BAIXANDO TOMOGRAFIAS DE CÂNCER DE PULMÃO DA INTERNET ===")
    arquivos_baixados = {}

    # Fonte 1: Repositório público de TC clínica (4 categorias)
    repo = "hallowshaw/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning"
    branch = "main"
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"

    try:
        resp = requests.get(tree_url, headers=HEADERS, timeout=20)
        tree_data = resp.json().get("tree", [])
    except Exception as e:
        print(f"Aviso ao consultar API do GitHub: {e}")
        tree_data = []

    categorias = [
        "adenocarcinoma",
        "large.cell.carcinoma",
        "squamous.cell.carcinoma",
        "normal"
    ]

    for cat in categorias:
        pasta_cat = TEST_IMAGES_DIR / cat.replace(".", "_")
        pasta_cat.mkdir(parents=True, exist_ok=True)
        arquivos_baixados[cat] = []

        # Pegar amostras diferentes (do final da lista para serem inéditas!)
        cat_files = [
            f["path"] for f in tree_data
            if f["path"].startswith(f"dataset/test/{cat}/")
            and f["path"].lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        selecionados = cat_files[-qtd_por_categoria:] if len(cat_files) >= qtd_por_categoria else cat_files

        for file_path in selecionados:
            nome_limpo = Path(file_path).name.replace(" ", "_").replace("(", "").replace(")", "")
            caminho_local = pasta_cat / nome_limpo
            if caminho_local.exists() and caminho_local.stat().st_size > 1000:
                arquivos_baixados[cat].append(caminho_local)
                continue

            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
            r = requests.get(raw_url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                caminho_local.write_bytes(r.content)
                arquivos_baixados[cat].append(caminho_local)
                print(f"  [OK] Baixada: {cat}/{nome_limpo}")

    # Fonte 2: Casos clínicos clássicos da Wikimedia Commons (Nódulos Pulmonares em TC)
    urls_extras = {
        "wikimedia_nodulo_solitario_01.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b2/Pulmonary_nodule_CT.jpg",
        "wikimedia_cancer_pulmao_tc_02.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Lung_cancer_CT.jpg",
        "wikimedia_adenocarcinoma_tc_03.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/CT_of_lung_adenocarcinoma.png/600px-CT_of_lung_adenocarcinoma.png",
    }
    pasta_extra = TEST_IMAGES_DIR / "casos_clinicos_internet"
    pasta_extra.mkdir(parents=True, exist_ok=True)
    arquivos_baixados["casos_clinicos_internet"] = []

    for nome_arq, url in urls_extras.items():
        caminho_local = pasta_extra / nome_arq
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                caminho_local.write_bytes(r.content)
                arquivos_baixados["casos_clinicos_internet"].append(caminho_local)
                print(f"  [OK] Baixada extra: {nome_arq}")
        except Exception:
            pass

    total_imgs = sum(len(v) for v in arquivos_baixados.values())
    print(f"\nTotal de imagens prontas para teste em '{TEST_IMAGES_DIR}': {total_imgs} imagens!\n")
    return arquivos_baixados


def encontrar_modelo_best_pt():
    """Procura automaticamente o arquivo best.pt na pasta Downloads ou na pasta do projeto."""
    downloads = Path.home() / "Downloads"
    candidatos = [
        BASE_DIR / "best.pt",
        BASE_DIR / "best_paciente_saudaveis.pt",
        downloads / "best.pt",
        downloads / "best_paciente_saudaveis.pt",
    ]
    # Também procura qualquer best*.pt recente em Downloads
    if downloads.exists():
        for p in sorted(downloads.glob("best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
            candidatos.append(p)

    for c in candidatos:
        if c.exists():
            return c
    return None


def rodar_testes_no_modelo(arquivos_baixados, caminho_modelo=None):
    if caminho_modelo is None:
        caminho_modelo = encontrar_modelo_best_pt()

    if caminho_modelo is None or not Path(caminho_modelo).exists():
        print("[AVISO] O arquivo 'best.pt' ainda nao foi encontrado na sua pasta Downloads nem na pasta do projeto.")
        print("        Assim que voce baixar o 'best.pt' do Google Colab para a pasta Downloads,")
        print("        basta rodar este script novamente que ele fara todas as deteccoes automaticamente!")
        return

    from ultralytics import YOLO
    import cv2
    import numpy as np

    print(f"=== 2. CARREGANDO MODELO YOLO: {caminho_modelo} ===")
    model = YOLO(str(caminho_modelo))

    resumo = {}
    for limiar in [0.25, 0.35]:
        pasta_limiar = RESULTS_DIR / f"conf_{int(limiar * 100)}"
        pasta_limiar.mkdir(parents=True, exist_ok=True)
        resumo[str(limiar)] = {}

        print(f"\n--- Testando com Limiar de Confiança conf = {limiar} ---")
        for cat, lista_arquivos in arquivos_baixados.items():
            if not lista_arquivos:
                continue
            detectados = 0
            confs = []

            for img_path in lista_arquivos:
                results = model.predict(source=str(img_path), conf=limiar, imgsz=224, verbose=False)
                res = results[0]
                n_boxes = len(res.boxes)
                if n_boxes > 0:
                    detectados += 1
                    for b in res.boxes:
                        confs.append(float(b.conf[0].cpu().numpy()))

                plotted = res.plot()
                out_file = pasta_limiar / f"{cat}_{img_path.name}"
                cv2.imwrite(str(out_file), plotted)

            taxa = (detectados / len(lista_arquivos)) * 100
            media_conf = float(np.mean(confs)) if confs else 0.0
            resumo[str(limiar)][cat] = {
                "total_imagens": len(lista_arquivos),
                "imagens_com_deteccao": detectados,
                "taxa_deteccao_pct": round(taxa, 1),
                "confianca_media": round(media_conf, 3),
            }
            print(f"  Categoria: {cat:26s} | Detectou: {detectados:2d}/{len(lista_arquivos):2d} ({taxa:5.1f}%) | Conf Média: {media_conf:.3f}")

    relatorio_json = RESULTS_DIR / "relatorio_testes_internet.json"
    relatorio_json.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Testes concluidos! Fotos anotadas e relatorio salvos em:\n   {RESULTS_DIR}")


if __name__ == "__main__":
    arquivos = baixar_imagens_internet(qtd_por_categoria=10)
    rodar_testes_no_modelo(arquivos)
