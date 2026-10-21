# Guia Completo do Treinamento de IA: Detecção Precoce de Nódulos Pulmonares (TCC)

> Documento consolidado com base no repositório [lealtais/TRAINING_IA_NODULES_LDC](https://github.com/lealtais/TRAINING_IA_NODULES_LDC) e nas avaliações do modelo `best.pt`.
>
> **Documentos Complementares no Repositório:**
> * **[`METODOLOGIA_TCC_ABNT.md`](METODOLOGIA_TCC_ABNT.md):** Texto formal acadêmico (padrão ABNT NBR 14724 / NBR 6023) detalhando a escolha e fusão das bases de dados, a justificativa matemática/estatística da divisão por paciente, proporção de amostras negativas e referências bibliográficas prontas para o TCC.
> * **[`GUIA_INFORMAL_ESTUDO_E_DUVIDAS.md`](GUIA_INFORMAL_ESTUDO_E_DUVIDAS.md):** Guia de bolso informal explicando todas as dúvidas sobre o funcionamento do YOLO, das pastas, dos arquivos `.txt`, da proporção dos bancos e do fluxo 100% remoto (Google Colab + GitHub) sem depender do PC local.
> * **[`train_lidc_por_paciente.py`](train_lidc_por_paciente.py):** Script completo pronto para rodar no Google Colab (divisão por paciente + adição de tomografias de pulmão saudável).

---

## 1. Visão Geral do Projeto

Este projeto consiste no desenvolvimento e avaliação de um modelo de **visão computacional para detecção precoce de nódulos pulmonares** em cortes de tomografia computadorizada (CT Scan), utilizando a arquitetura **YOLOv8** (Ultralytics).

* **Objetivo:** Localizar espacialmente o nódulo dentro do corte de TC através de *bounding boxes* (caixas delimitadoras), auxiliando o médico radiologista na triagem precoce, e não apenas emitir um rótulo binário se o paciente tem ou não câncer.
* **Classe Treinada:** `0: 'nodulo'`
* **Status:** Projeto em desenvolvimento para Trabalho de Conclusão de Curso (TCC).

---

## 2. Escolha e Metodologia de Datasets

Dois grandes conjuntos de dados médicos foram avaliados durante a pesquisa:

### A. DeepLesion (NIH) — *Descartado como dataset principal*
* **Origem:** Repositório do NIH (National Institutes of Health).
* **Amostra avaliada:** `kmader/nih-deeplesion-subset` (Kaggle), com 1.327 imagens aproveitáveis.
* **Motivo do descarte:** O DeepLesion cataloga lesões em múltiplos órgãos (fígado, rim, pâncreas, pulmão, linfonodos, etc.). Como a meta do TCC é específica para nódulos de pulmão, o modelo não generalizou satisfatoriamente.
* **Resultado do treino (YOLOv8s, 100 épocas):** `mAP50 = 0.329` | `Precisão = 0.557` | `Recall = 0.304`.

### B. LIDC-IDRI-slices — *Dataset Escolhido*
* **Origem:** `zhangweiled/lidcidri` (Kaggle), referência internacional em rastreamento de câncer pulmonar.
* **Volume:** 875 pacientes, totalizando 15.548 cortes axiais de TC (15.486 contendo nódulos confirmados).
* **Decisão Metodológica Crucial (Fusão de Radiologistas):** Cada nódulo no LIDC-IDRI possui máscaras anotadas por 4 radiologistas independentes. Para garantir robustez e evitar viés de um único especialista, foi aplicada a **união lógica das 4 máscaras**: se pelo menos 1 radiologista assinalou o pixel como nódulo, ele é incorporado à bounding box mínima.

---

## 3. Histórico de Treinamentos e Métricas

| Modelo | Dataset | Épocas | imgsz | Otimizador | Precisão | Recall | mAP@50 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n** | DeepLesion | 30 | 512 | Padrão | 31.7% | 24.6% | 20.8% | Descartado |
| **YOLOv8s** | DeepLesion | 100 | 512 | Padrão | 55.7% | 30.4% | 32.9% | Descartado |
| **YOLOv8s** | **LIDC-IDRI** | **50** | **224** | **AdamW** | **80.9%** | **71.2%** | **75.4%** | **Melhor Modelo (`best.pt`)** |

> **Conclusão:** A transição para um dataset dedicado exclusivamente a nódulos pulmonares (LIDC-IDRI) com fusão de máscaras proporcionou um salto expressivo de desempenho: o **mAP@50 subiu de 32.9% para 75.4%**.

---

## 4. Curvas de Aprendizado do Modelo (`best.pt`)

As métricas registradas durante as 50 épocas de treinamento do modelo final revelam convergência estável:

![Curvas de Treinamento e Validação](assets/training_curves.png)

### Análise das Curvas:
1. **Box Loss (Perda de Localização):** Queda contínua tanto no conjunto de treino quanto no de validação, comprovando que o modelo aprendeu a ajustar o retângulo ao redor do nódulo sem overfitting severo.
2. **Classification Loss:** Estabilizada em valores muito baixos desde as primeiras épocas.
3. **Evolução do mAP:** O `mAP@50` ultrapassou 70% já próximo à 25ª época, estabilizando-se em **75.4%**.
4. **Precisão vs Recall:** Equilíbrio sólido entre Precisão (~81%) e Recall (~71%).

---

## 5. Problemas Técnicos e Lições Aprendidas

1. **Bug do Otimizador Muon no Ultralytics:**
   * *Erro:* `RuntimeError: view size is not compatible...` dentro de `optim/muon.py`.
   * *Solução:* Fixar explicitamente o otimizador no treinamento com `optimizer="AdamW"`.
2. **Falso Salvamento de Checkpoint no Google Colab:**
   * *Incidente:* O script executava o comando de cópia para `/content/drive/MyDrive/...`, mas como o `drive.mount()` não havia sido autenticado na sessão, os pesos só existiam no disco efêmero e foram perdidos ao encerrar a máquina.
   * *Lição:* Sempre checar se a montagem do Drive foi confirmada pelo popup e verificar se o arquivo realmente consta na nuvem antes de fechar a VM.

---

## 6. Avaliação em Casos Reais e Calibração de Confiança (`conf`)

Para simular o uso em um ambiente hospitalar, o modelo foi testado em 40 tomografias clínicas divididas entre patologias malignas e pulmões normais:

| Limiar (`conf`) | Adenocarcinoma (10) | Grandes Células (10) | Células Escamosas (10) | Pulmão Sadio (10) | Falsos Positivos |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 7 / 10 | 10 / 10 | 7 / 10 | 10 / 10 | 100% |
| **0.20** | 6 / 10 | 6 / 10 | 7 / 10 | 10 / 10 | 100% |
| **0.25** | 5 / 10 | 6 / 10 | 7 / 10 | 8 / 10 | 80% |
| **0.30** | 3 / 10 | 6 / 10 | 6 / 10 | 2 / 10 | 20% |
| **0.35 (Recomendado)** | **3 / 10** | **6 / 10** | **6 / 10** | **0 / 10** | **0.0% (Especificidade Máxima)** |
| **0.40** | 3 / 10 | 5 / 10 | 4 / 10 | 0 / 10 | 0.0% |

### Visualização com Limiar Ótimo (`conf = 0.35`):
Com este corte de confiança, o modelo detecta as lesões primárias com confianças de até 76% e elimina qualquer falso alarme em tecidos sadios.

![Predições com Limiar Ótimo conf=0.35](assets/optimal_threshold_conf35.png)

### Visualização com Limiar Permissivo (`conf = 0.25`):
Neste limiar mais baixo, há maior sensibilidade, porém estruturas anatômicas normais começam a ser marcadas:

![Predições com Limiar conf=0.25](assets/test_inference_samples.png)

---

## 7. Testes de Inferência Externa (Imagens do Repositório)

Testes realizados fora do dataset de treinamento para avaliar a generalização:

### Teste 01 — Nódulos Múltiplos e Miliares
* **Resultado:** 0 detecções.
* **Diagnóstico:** Padrão de múltiplos micronódulos difusos em ambos os pulmões. O modelo foi treinado predominantemente com nódulos isolados e bem demarcados do LIDC-IDRI, evidenciando uma oportunidade de melhoria para nódulos sub-centimétricos.

| Imagem Original | Predição do Modelo |
| :---: | :---: |
| ![Original 01](assets/teste_01_original.jpg) | ![Detecção 01](assets/teste_01_deteccao.jpg) |

---

### Teste 02 — Corte Axial Típico
* **Resultado:** 2 detecções (Confiança: 0.70 e 0.36).
* **Diagnóstico:** Excelente identificação do nódulo principal na periferia pulmonar.

| Imagem Original | Predição do Modelo |
| :---: | :---: |
| ![Original 02](assets/teste_02_original.jpg) | ![Detecção 02](assets/teste_02_deteccao.jpg) |

---

### Teste 03 — Raio-X de Tórax (Modalidade Diferente)
* **Resultado:** 0 detecções.
* **Diagnóstico Correto:** Este teste foi crucial metodologicamente. Como o modelo foi treinado estritamente com tomografia computadorizada (cortes axiais 2D seccionais), ele não deve gerar previsões espúrias em Raio-X (projeção torácica 2D). O resultado zero comprova que o modelo não "alucina" caixas em modalidades não suportadas.

| Imagem Original | Predição do Modelo |
| :---: | :---: |
| ![Original 03](assets/teste_03_original.png) | ![Detecção 03](assets/teste_03_deteccao.jpg) |

---

### Teste 04 — Novo Corte de TC
* **Resultado:** 1 detecção (Confiança: 0.29).
* **Diagnóstico:** Nódulo sutil detectado; confiança menor reflete atenuação e contraste reduzido na lesão.

| Imagem Original | Predição do Modelo |
| :---: | :---: |
| ![Original 04](assets/teste_04_original.png) | ![Detecção 04](assets/teste_04_deteccao.jpg) |

---

## 8. Testes Secundários com o Modelo DeepLesion

Investigação sobre o comportamento do modelo generalista DeepLesion em órgãos não pulmonares:

| Teste | Órgão / Região | Resultado | Observações |
| :---: | :---: | :---: | :--- |
| **01** | Pâncreas / Fígado | 0 detecções | Não identificou lesão |
| **02** | Abdômen (Pâncreas / Rim) | 3 detecções | Confianças baixas (0.52 / 0.19 / 0.16) e uma caixa degenerada |
| **03** | Pulmão (mesmo corte do Teste 04) | 2 detecções | Confiança baixa (0.17 / 0.16), inferior ao modelo LIDC-IDRI |

### Comparações Visuais (DeepLesion):

#### Pâncreas / Fígado:
| Original | Detecção |
| :---: | :---: |
| ![DL 01 Original](assets/dl_teste_01_original.png) | ![DL 01 Detecção](assets/dl_teste_01_deteccao.jpg) |

#### Abdômen:
| Original | Detecção |
| :---: | :---: |
| ![DL 02 Original](assets/dl_teste_02_original.png) | ![DL 02 Detecção](assets/dl_teste_02_deteccao.jpg) |

#### Pulmão (Comparativo com LIDC-IDRI):
| Original | Detecção |
| :---: | :---: |
| ![DL 03 Original](assets/dl_teste_03_original.png) | ![DL 03 Detecção](assets/dl_teste_03_deteccao.jpg) |

---

## 9. Limitações Identificadas e Planejamento (Próximos Passos)

### 9.1 Análise Crítica e Limitações Atuais
1. **Divisão por Fatia vs. Divisão por Paciente (*Data Leakage*):**
   * O resultado atual de **mAP@50 = 0.754** provavelmente está inflado porque a divisão entre treino e validação foi feita sorteando fatias individuais, e não pacientes. Como um mesmo nódulo aparece em várias fatias vizinhas (quase idênticas), fatias irmãs do mesmo nódulo caíram tanto no treino quanto na validação. Isso explica em parte por que o desempenho caiu nos testes com imagens externas.
2. **Gargalo em Variedade e Avaliação Honesta (e não em volume bruto):**
   * O dataset bruto de 38 GB (`lidcidri30`, com arquivos DICOM e XML) vem da mesma fonte (LIDC-IDRI), portanto não traz padrões novos que faltam ao modelo atual, como o padrão miliar (vários micronódulos difusos).
3. **Tamanho da Amostra de Testes Externos:**
   * Os testes iniciais com 4 a 5 imagens da internet permitiram levantar ótimas hipóteses (acerto em nódulos isolados, falha no padrão miliar e ausência esperada de detecção em raio-X), mas precisam ser ampliados para conclusões definitivas.

### 9.2 Próximos Passos (Em Ordem de Prioridade)
1. **Refazer o treino com divisão por paciente (`train_lidc_por_paciente.py`):** Usar o dataset `zhangweiled/lidcidri` garantindo que todas as fatias de um paciente fiquem 100% no treino ou 100% na validação. É rápido, aproveita o pipeline que já funciona e entrega o baseline honesto para a banca do TCC.
2. **Adicionar imagens sem nódulo (pulmão saudável):** Incluir fatias negativas no treinamento para ensinar o modelo o que é tecido normal e reduzir falsos positivos.
3. **Ampliar os testes externos:** Testar com um conjunto maior de imagens independentes, documentando cada caso no log de experimentos.
4. **Atacar a falha dos nódulos múltiplos pequenos:** Buscar dados complementares de outra fonte que contenham padrão miliar / micronódulos múltiplos.
5. **Usar o `lidcidri30` (38 GB) apenas se sobrar tempo:** Aproveitar principalmente como fonte de imagens negativas (sem nódulo), evitando perder tempo excessivo na conversão complexa de XML e DICOM.
6. **Exportar para TFLite e integrar no app:** Realizar a conversão final e integração na aplicação depois que o modelo com divisão por paciente estiver consolidado.
