# FUNDAMENTAÇÃO METODOLÓGICA, SELEÇÃO DE BASES DE DADOS E ARQUITETURA DE TREINAMENTO (PADRÃO ABNT)

> **Nota de Utilização Acadêmica:** Este documento foi redigido segundo as normas de redação científica e referenciação da **ABNT (NBR 14724 e NBR 6023)**, podendo ser incorporado diretamente ao capítulo de **Materiais e Métodos (Metodologia)** e **Discussão** do Trabalho de Conclusão de Curso (TCC).

---

## 3 MATERIAIS E MÉTODOS

### 3.1 Definição do Escopo Clínico e Computacional

O presente trabalho desenvolve um sistema de auxílio ao diagnóstico por computador (*Computer-Aided Detection* — CADe) voltado à **localização espacial precoce de nódulos pulmonares** em cortes axiais de Tomografia Computadorizada (TC) de tórax, empregando a arquitetura de detecção de objetos em estágio único **YOLOv8s** (*You Only Look Once*, versão 8 *small*) (JOCHER; CHAURASIA; QIU, 2023).

Diferentemente de abordagens baseadas exclusivamente em classificação binária em nível de imagem — as quais apenas indicam a presença ou ausência global de patologia sem informar sua topografia anatômica —, a formulação adotada baseia-se na regressão de caixas delimitadoras (*bounding boxes*) parametrizadas pelo vetor $(x_c, y_c, w, h)$, permitindo apontar diretamente ao médico radiologista a região suspeita dentro do parênquima pulmonar. O escopo restringe-se à **detecção e localização morfológica** da classe única `0: nodulo`, atuando na triagem clínica inicial e não substituindo a biópsia ou o laudo histopatológico quanto à malignidade da lesão.

---

### 3.2 Seleção, Avaliação e Fusão das Bases de Dados

A construção de um conjunto de dados (*dataset*) representativo para detecção de lesões médicas exige especificidade anatômica, rigor de anotação especialista e presença de amostras de controle (exames normais). Ao longo do desenvolvimento experimental, quatro bases de dados públicas foram avaliadas e integradas segundo critérios estritos de engenharia de dados médicos.

#### 3.2.1 Descarte da Base Generalista Multi-Órgãos (*NIH DeepLesion*)

Inicialmente, investigou-se o subconjunto público *NIH DeepLesion* (YAN et al., 2018), composto por cortes tomográficos de múltiplas regiões anatômicas. Após a filtragem de integridade entre metadados e arquivos de imagem, obtiveram-se 1.327 cortes utilizáveis. Contudo, por tratar-se de um repositório generalista que engloba lesões hepáticas, renais, pancreáticas, abdominais, linfonodais e pulmonares sob uma mesma estrutura de anotação, o treinamento com a arquitetura YOLOv8s por 100 épocas resultou em desempenho insuficiente para aplicação torácica dedicada, atingindo precisão (*Precision*) de $0,557$, revocação (*Recall*) de $0,304$ e média de precisão média (*mAP@50*) de $0,329$.

Constatou-se que a heterogeneidade anatômica diluiu o aprendizado dos filtros convolucionais para o parênquima pulmonar, justificando o descarte fundamentado do *DeepLesion* como base principal do projeto.

#### 3.2.2 Adoção da Base Especializada *LIDC-IDRI* e Fusão por Consenso de Quatro Radiologistas

Como base principal de lesões positivas, selecionou-se o repositório *Lung Image Database Consortium and Image Database Resource Initiative* (**LIDC-IDRI**) (ARMATO III et al., 2011), em sua distribuição pré-processada em cortes axiais (*LIDC-IDRI-slices*, disponibilizada por `zhangweiled/lidcidri`), abrangendo **875 pacientes** e **15.548 cortes tomográficos** (dos quais **15.486** apresentam máscaras válidas de nódulos pulmonares e 62 cortes com máscaras vazias decorrentes de artefatos de borda foram descartados).

Uma característica metodológica fundamental do LIDC-IDRI é que cada nódulo foi segmentado de forma independente por até **quatro médicos radiologistas torácicos**, gerando quatro máscaras binárias ($M_0, M_1, M_2, M_3$). Na literatura médica, sabe-se que há expressiva variabilidade interobservador na delimitação de bordas nodulares (ARMATO III et al., 2011). Para mitigar o viés individual de um único especialista e priorizar a sensibilidade clínica de rastreamento, implementou-se a **união lógica booleana (operador OR)** das quatro máscaras:

$$M_{\text{consenso}}(x, y) = \bigcup_{k=0}^{3} M_k(x, y) = M_0(x, y) \lor M_1(x, y) \lor M_2(x, y) \lor M_3(x, y)$$

Dessa forma, qualquer pixel $(x, y)$ assinalado com intensidade superior ao limiar binário ($> 127$) por ao menos um dos quatro radiologistas é incorporado à região nodular consolidada. A partir da matriz resultante $M_{\text{consenso}}$, extraem-se os limites extremos $(x_{\min}, y_{\min})$ e $(x_{\max}, y_{\max})$, convertendo-os para o formato normalizado exigido pela arquitetura YOLO em relação à largura ($W$) e altura ($H$) da imagem:

$$x_c = \frac{x_{\min} + x_{\max}}{2W}, \quad y_c = \frac{y_{\min} + y_{\max}}{2H}, \quad w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$

#### 3.2.3 Integração de Bases Complementares de Controle Negativo (Pulmões Saudáveis)

Uma limitação estrutural da distribuição *LIDC-IDRI-slices* é que a totalidade de seus diretórios válidos refere-se a recortes contendo nódulos confirmados (amostras positivas). Quando uma rede neural detectora de objetos é submetida ao treinamento exclusivamente com imagens positivas, induz-se um viés probabilístico no qual o modelo pressupõe a existência obrigatória de ao menos uma lesão em todo exame analisado, elevando a taxa de **falsos positivos** sobre estruturas anatômicas normais hiperatenuantes, tais como bifurcações vasculares hilares, ramos brônquicos e arcos costais.

Para solucionar esse problema sem incorrer no elevado custo computacional e de armazenamento da extração bruta de arquivos DICOM/XML do repositório LIDC-IDRI original (38 GB), integraram-se ao conjunto de treinamento as coortes de controle saudável (*Normal cases*) provenientes de duas bases públicas revisadas por especialistas:
1. **The IQ-OTH/NCCD Lung Cancer Dataset** (AL-YASRIY et al., 2020), coletada no *Iraq Oncology Teaching Hospital / National Center for Cancer Diseases*, fornecendo cortes axiais de pacientes com parênquima pulmonar comprovadamente hígido;
2. **Chest CT-Scan Images Dataset** (`mohamedhanyyy/chest-ctscan-images`), fornecendo cortes adicionais de controle tomográfico normal.

Ao todo, incorporaram-se mais de **630 tomografias de pulmões saudáveis (sem nódulos)** como amostras negativas (*background images*).

---

### 3.3 Esclarecimentos Metodológicos e Decisões de Engenharia de Dados

Nesta seção, formalizam-se as justificativas técnicas para as decisões de particionamento, balanceamento e estruturação dos dados no pipeline de treinamento.

#### 3.3.1 Particionamento em Nível de Paciente (*Patient-Level Split*) versus Nível de Fatia (*Slice-Level Split*)

Na tomografia computadorizada helicoidal/multislice, o tórax do paciente é adquirido através de múltiplos cortes axiais contíguos separados por poucos milímetros. Consequentemente, um único nódulo pulmonar de um mesmo paciente manifesta-se em diversas fatias sequenciais adjacentes (por exemplo, fatias $i, i+1, i+2$), as quais apresentam altíssima correlação espacial e estrutural entre si.

Quando a divisão entre os conjuntos de treinamento (*train*) e validação (*val*) é realizada mediante sorteio aleatório de fatias individuais (*slice-level split*), ocorre o fenômeno conhecido na literatura de aprendizado de máquina aplicado à medicina como **vazamento de dados intra-paciente (*intra-patient data leakage*)** (BUSSOLA et al., 2021): a fatia $i$ de determinado paciente é alocada no conjunto de treinamento, ao passo que a fatia adjacente $i+1$ (quase idêntica) do mesmo indivíduo é alocada no conjunto de validação. Esse procedimento infla artificialmente as métricas de validação (como observado no experimento preliminar com *mAP@50* de $0,754$), pois o modelo é avaliado em anatomias e lesões que já memorizou parcialmente durante o treinamento, degradando sua capacidade real de generalização perante pacientes inéditos.

Para garantir rigor científico e validade clínica, implementou-se o **particionamento estrito em nível de paciente (*patient-level split*)**:
1. Listam-se inicialmente os identificadores únicos dos **875 pacientes** (`LIDC-IDRI-0001` a `LIDC-IDRI-0875`);
2. Aplica-se a função de particionamento estratificado (`train_test_split`, com semente reprodutível `random_state = 42` e proporção de $85\% / 15\%$) diretamente sobre o vetor de pacientes, destinando **743 pacientes exclusivamente para treinamento** e **132 pacientes exclusivamente para validação**;
3. Impõe-se uma verificação formal de disjunção de conjuntos ($\text{Pacientes}_{\text{treino}} \cap \text{Pacientes}_{\text{val}} = \emptyset$), assegurando que todas as fatias e todos os nódulos de um mesmo indivíduo pertençam integralmente a apenas uma das partições.

#### 3.3.2 Justificativa da Proporção entre Fatias com Nódulo (~15.486) e Imagens de Pulmão Saudável (~630)

Um questionamento recorrente na modelagem de dados refere-se à aparente diferença numérica entre os **15.486 cortes positivos** (provenientes dos 875 pacientes do LIDC-IDRI) e os **~630 cortes negativos** de pulmão saudável adicionados. Cumpre esclarecer tecnicamente por que tal proporção **não configura desbalanceamento prejudicial** na arquitetura YOLOv8:

1. **Diferença entre Classificação de Imagem e Detecção Densa por Âncoras/Células:** Em redes classificadoras convencionais (ex.: ResNet, VGG), cada imagem representa um único voto global (1 exemplo positivo ou 1 exemplo negativo), exigindo proporções próximas a $50\% / 50\%$. Em contrapartida, detectores da família YOLO dividem cada imagem em milhares de células de grade (*grid cells*) e pontos de predição espacial. Mesmo dentro de uma imagem positiva (com nódulo), cerca de **$95\%$ a $98\%$ da área da imagem corresponde a tecido pulmonar sadio de fundo (*background*)**, e apenas $2\%$ a $5\%$ dos pixels correspondem à caixa delimitadora do nódulo. Portanto, o modelo já processa milhões de amostras locais de tecido não-nodular a cada época.
2. **Diretriz Oficial da Arquitetura YOLOv8:** Segundo a documentação técnica oficial da Ultralytics (JOCHER; CHAURASIA; QIU, 2023), a proporção recomendada de imagens integralmente negativas (*background images*, isentas de objetos) situa-se na faixa de **$1\%$ a $10\%$ do total de imagens do dataset**. A adição de ~630 imagens de pulmão saudável perante ~15.486 fatias positivas representa aproximadamente **$4\%$ do volume total de imagens**, enquadrando-se exatamente na faixa ótima recomendada pela engenharia do algoritmo para suprimir falsos positivos sem reduzir a sensibilidade (*Recall*) do detector.

#### 3.3.3 Harmonização de Domínio na Fusão de Múltiplas Bases (*Domain Harmonization*)

Ao combinar imagens oriundas de repositórios distintos (*LIDC-IDRI*, *IQ-OTH/NCCD* e *Chest CT-Scan Images*), deve-se prevenir que a rede neural aprenda características espúrias de codificação de arquivo (como número de canais de cor RGB vs. escala de cinza) como atalho decisório (*domain shortcut*). Para assegurar homogeneidade radiométrica:
* Todas as imagens provenientes das bases de pulmão saudável são explicitamente convertidas para matrizes monocromáticas de 8 bits em escala de cinza (`modo "L"`, 1 canal de luminância) e padronizadas em formato `.png` sem perdas antes da etapa de treinamento, espelhando fielmente a estrutura de canais dos cortes do *LIDC-IDRI*.

#### 3.3.4 Estrutura de Diretórios e Pareamento de Rótulos no Padrão YOLO

Diferentemente de arquiteturas de classificação que separam classes em subpastas distintas (`com_nodulo/` vs. `sem_nodulo/`), o protocolo de dados do YOLOv8 organiza os arquivos estritamente por partição experimental (**treinamento** e **validação**), mantendo pastas rigorosamente separadas para `train` e `val`:

```text
yolo/
 ├── images/
 │    ├── train/   (Imagens .png dos 743 pacientes de treino + 85% dos pulmões saudáveis)
 │    └── val/     (Imagens .png dos 132 pacientes de validação + 15% dos pulmões saudáveis)
 └── labels/
      ├── train/   (Arquivos .txt espelho de mesmo nome para cada imagem de treino)
      └── val/     (Arquivos .txt espelho de mesmo nome para cada imagem de validação)
```

A distinção entre uma tomografia contendo nódulo e uma tomografia de pulmão saudável dentro de cada partição (`train` ou `val`) é efetuada de maneira determinística pelo conteúdo do arquivo `.txt` homônimo localizado no diretório `labels/`:
* **Amostra Positiva (Com Nódulo):** O arquivo `.txt` correspondente contém a linha parametrizada `<classe> <x_center> <y_center> <width> <height>` (ex.: `0 0.512000 0.430000 0.080000 0.075000`), instruindo a função de custo (*CIoU Box Loss* e *Binary Cross-Entropy Classification Loss*) a regredir a caixa delimitadora sobre a lesão.
* **Amostra Negativa (Pulmão Saudável / Sem Nódulo):** O arquivo `.txt` correspondente é gerado **integralmente vazio ($0\text{ bytes}$)**. Durante a retropropagação (*backpropagation*), a ausência de coordenadas no arquivo `.txt` informa ao modelo que a expectativa de detecção naquela imagem é nula ($\hat{y} = 0$), penalizando qualquer predição de caixa delimitadora e ensinando a rede a não emitir falsos alarmes diante de vasos pulmonares e estruturas normais do mediastino.

---

### 3.4 Infraestrutura de Treinamento em Nuvem e Reprodutibilidade Remota

Tendo em vista a natureza intensiva das operações matriciais convolucionais sobre mais de 16.000 cortes tomográficos ao longo de 50 épocas — inviável em tempo hábil mediante processamento exclusivo em Unidade Central de Processamento (CPU) local —, todo o fluxo de processamento de dados, download automatizado via API (`kagglehub`) e treinamento com aceleração por hardware em Unidade de Processamento Gráfico (**GPU NVIDIA Tesla T4**) foi estruturado para execução integralmente em nuvem (ambiente *Google Colaboratory*), sincronizado ao repositório remoto de controle de versão no GitHub (`lealtais/TRAINING_IA_NODULES_LDC`). Essa arquitetura garante reprodutibilidade científica independente de hardware local.

---

## REFERÊNCIAS (PADRÃO ABNT NBR 6023)

AL-YASRIY, H. F. et al. Diagnosis of lung cancer based on CT scans using convolutional neural networks. **IOP Conference Series: Materials Science and Engineering**, Bristol, v. 928, n. 2, p. 022035, nov. 2020.

ARMATO III, S. G. et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. **Medical Physics**, Hoboken, v. 38, n. 2, p. 915-931, fev. 2011.

BUSSOLA, N. et al. AI slipping on tiles: data leakage in digital pathology. In: INTERNATIONAL CONFERENCE ON PATTERN RECOGNITION (ICPR), 25., 2021, Milan. **Proceedings [...]**. Cham: Springer, 2021. p. 167-182.

JOCHER, G.; CHAURASIA, A.; QIU, J. **Ultralytics YOLOv8**. Versão 8.0.0. [S. l.]: Ultralytics, 2023. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 20 out. 2026.

YAN, K. et al. DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. **Journal of Medical Imaging**, Bellingham, v. 5, n. 3, p. 036501, jul. 2018.
