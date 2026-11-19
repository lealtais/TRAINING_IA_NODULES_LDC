# Documentação para Submissão na Plataforma Brasil (CEP / CONEP)

Pasta com todos os textos prontos (copiar e colar) e os documentos em PDF exigidos para a submissão do TCC na **Plataforma Brasil** (`https://plataformabrasil.saude.gov.br`).

---

## 📄 Documentos em PDF Prontos para Upload (Etapa 6 da Plataforma Brasil)

1. **[`01_PROJETO_DETALHADO_PLATAFORMA_BRASIL.pdf`](01_PROJETO_DETALHADO_PLATAFORMA_BRASIL.pdf)** — Projeto de pesquisa completo estruturado no padrão do Comitê de Ética (CEP), já contemplando os **3 bancos públicos anonimizados** (`LIDC-IDRI`, `IQ-OTH/NCCD` e `Chest CT-Scan Images`) e deixando explícito que o cadastro de paciente no aplicativo é apenas proposta arquitetural testada com dados fictícios.
2. **[`02_SOLICITACAO_DISPENSA_TCLE.pdf`](02_SOLICITACAO_DISPENSA_TCLE.pdf)** — Carta formal de pedido de **Dispensa do Termo de Consentimento Livre e Esclarecido (TCLE)** (basta assinar e subir).
3. **[`03_TERMO_COMPROMISSO_USO_DADOS_TCUD.pdf`](03_TERMO_COMPROMISSO_USO_DADOS_TCUD.pdf)** — **Termo de Compromisso de Utilização de Dados (TCUD)** (frequentemente exigido pelos CEPs em pesquisas com bases secundárias públicas).
4. **[`04_CRONOGRAMA_E_ORCAMENTO.pdf`](04_CRONOGRAMA_E_ORCAMENTO.pdf)** — Cronograma detalhado e Declaração de Orçamento (R$ 0,00 — recursos próprios / ambiente em nuvem gratuito).

> **Lembrete:** A **Folha de Rosto** é gerada automaticamente pela própria Plataforma Brasil na Etapa 1. Você baixa, colhe a assinatura do Orientador / Direção da FATEC Rubens Lara, escaneia em PDF e faz o upload na Etapa 6 junto com os 4 PDFs acima.

---

## 📋 Textos Prontos para Copiar e Colar nas 6 Etapas do Formulário

### Etapa 1 — Informações Preliminares
* **Título Público da Pesquisa:** Inteligência Artificial: suporte ao diagnóstico precoce do câncer de pulmão
* **Instituição Proponente:** Faculdade de Tecnologia Baixada Santista Rubens Lara (FATEC Rubens Lara / Centro Paula Souza)
* **Pesquisador Responsável:** *(Obs.: Muitos CEPs exigem que o **Orientador — Prof. João Paulo de Melo** seja cadastrado como Pesquisador Responsável e a aluna **Taís Leal Almeida** como Pesquisadora Assistente / Equipe de Pesquisa. Confirme com o Prof. João Paulo antes de finalizar a Etapa 1!)*

---

### Etapa 3 — Desenho do Estudo e Apoio Financeiro
* **Desenho:** Estudo observacional, retrospectivo e computacional baseado exclusivamente em bases de dados públicas secundárias e previamente anonimizadas de imagens médicas (Tomografia Computadorizada de tórax), sem contato direto ou indireto com participantes humanos.
* **Apoio Financeiro:** Financiamento próprio (R$ 0,00 — uso de infraestrutura computacional acadêmica e ambiente gratuito em nuvem Google Colaboratory).

---

### Etapa 4 — Detalhamento do Estudo (Copiar e Colar nos Campos)

#### Resumo
> O câncer de pulmão figura como a principal causa de morte por neoplasia no mundo, sendo o diagnóstico tardio o maior fator associado à baixa taxa de sobrevida. Este Trabalho de Conclusão de Curso (TCC) tem como objetivo desenvolver e avaliar um modelo de Inteligência Artificial baseado em Redes Neurais Convolucionais (arquitetura YOLOv8s) para auxiliar médicos radiologistas na detecção precoce e localização espacial de nódulos pulmonares em imagens de Tomografia Computadorizada (TC) de tórax. A pesquisa utiliza exclusivamente imagens retrospectivas, públicas e integralmente anonimizadas provenientes de três repositórios internacionais de acesso aberto: LIDC-IDRI (875 pacientes, 15.362 recortes com nódulos anotados por radiologistas), IQ-OTH/NCCD e Chest CT-Scan Images (631 cortes de pulmões saudáveis utilizados como controle negativo para redução de falsos positivos), adotando particionamento estrito em nível de paciente (85% treino / 15% validação). Adicionalmente, propõe-se um protótipo de aplicação multiplataforma (Web e mobile) integrada via API em Python como ferramenta de apoio à decisão clínica. Destaca-se que não haverá recrutamento, entrevista ou coleta de exames de pacientes reais em instituições hospitalares; quaisquer funcionalidades de cadastro ou prontuário no protótipo do sistema serão testadas estritamente com dados fictícios/simulados.

#### Introdução
> Segundo dados da Agência Internacional de Pesquisa em Câncer (IARC/GLOBOCAN), o câncer de pulmão foi responsável por mais de 1,8 milhão de óbitos em 2020. Em decorrência de sua evolução inicial silenciosa e assintomática, grande parte dos casos é diagnosticada em estágios avançados, quando as taxas de sobrevida em cinco anos caem drasticamente em comparação ao Estágio I (SEER). Diante da alta demanda hospitalar e da suscetibilidade humana à fadiga visual na triagem de múltiplos cortes tomográficos, sistemas de auxílio ao diagnóstico por computador (Computer-Aided Detection — CADe) baseados em Deep Learning despontam como ferramentas promissoras de segunda opinião computacional.

#### Hipótese
> O treinamento de uma rede neural detectora de objetos (YOLOv8s) com particionamento estrito por paciente, combinando recortes anotados por múltiplos radiologistas (LIDC-IDRI) com cortes de controle saudável (IQ-OTH/NCCD e Chest CT-Scan Images), permite localizar precocemente nódulos pulmonares em tomografias computadorizadas com elevada precisão e redução expressiva de falsos positivos em exames normais, atuando como ferramenta segura de suporte à triagem médica.

#### Objetivo Primário
> Desenvolver, treinar e validar um modelo de Inteligência Artificial baseado na arquitetura YOLOv8s para auxiliar a equipe médica na detecção precoce e localização espacial de nódulos pulmonares em cortes de Tomografia Computadorizada de tórax, utilizando exclusivamente bases públicas anonimizadas.

#### Objetivos Secundários
> 1. Estruturar um pipeline de pré-processamento e fusão por consenso das anotações de múltiplos radiologistas da base LIDC-IDRI, aplicando particionamento estrito em nível de paciente para evitar vazamento de dados (data leakage);
> 2. Integrar amostras de controle negativo (pulmões saudáveis) oriundas das bases públicas anonimizadas IQ-OTH/NCCD e Chest CT-Scan Images para minimizar taxas de falsos positivos;
> 3. Avaliar o desempenho do modelo treinado mediante métricas padronizadas de visão computacional (Precisão, Recall, mAP@50, mAP@50-95 e Matriz de Confusão);
> 4. Disponibilizar o modelo por meio de uma API em Python integrada a um protótipo multiplataforma (Web/mobile) validado exclusivamente com dados simulados/fictícios.

#### Metodologia Proposta
> Trata-se de uma pesquisa aplicada e experimental em Ciência da Computação (Visão Computacional), conduzida integralmente em ambiente computacional com dados secundários públicos e anonimizados. Serão utilizados três repositórios internacionais de acesso público: (1) LIDC-IDRI-slices (875 pacientes, 15.362 recortes com nódulos); (2) The IQ-OTH/NCCD Lung Cancer Dataset (cortes normais); e (3) Chest CT-Scan Images (cortes normais), totalizando 15.993 imagens em escala de cinza (formato PNG). Os pacientes do LIDC-IDRI serão divididos em 85% para treinamento (743 pacientes) e 15% para validação (132 pacientes), garantindo disjunção total de indivíduos. O treinamento será realizado com o modelo YOLOv8s em ambiente de nuvem (Google Colaboratory). O protótipo de software (API em Python + interface Web/mobile) previsto no TCC tem caráter de demonstração arquitetural para uso futuro por profissionais médicos; durante toda a pesquisa, não serão inseridos dados pessoais, nomes, CPFs ou prontuários de pacientes reais, empregando-se exclusivamente registros fictícios nos testes de interface.

#### Critérios de Inclusão
> Cortes axiais de Tomografia Computadorizada de tórax disponibilizados publicamente e previamente anonimizados nos repositórios internacionais LIDC-IDRI, IQ-OTH/NCCD e Chest CT-Scan Images, contendo máscaras de segmentação válidas de nódulos pulmonares (grupo positivo) ou rotulagem confirmada de parênquima pulmonar normal (grupo controle negativo).

#### Critérios de Exclusão
> Imagens provenientes dos referidos bancos públicos que apresentem arquivos corrompidos, ausência de correspondência de metadados ou máscaras de anotação integralmente vazias no subconjunto positivo (62 fatias descartadas na limpeza do LIDC-IDRI), bem como imagens de outras modalidades radiológicas (como radiografias convencionais / Raio-X).

#### Riscos
> Por tratar-se de estudo exclusivamente computacional com dados secundários de acesso público e previamente desidentificados (anonimizados na origem pelos respectivos consórcios internacionais), não há qualquer risco físico, biológico, psicológico ou de intervenção clínica a seres humanos. O risco residual teórico de reidentificação de indivíduos ou de uso indevido do software como substituto do diagnóstico médico será mitigado mediante: (i) manutenção estrita da anonimização original sem qualquer tentativa de rastreio de identidade; (ii) ausência total de coleta de dados pessoais reais; e (iii) explicitação clara na interface do protótipo de que o sistema possui caráter exclusivamente acadêmico e de auxílio à decisão médica, não substituindo o laudo radiológico ou histopatológico.

#### Benefícios
> Não há benefícios diretos individuais aos pacientes cujas imagens compõem os bancos públicos retrospectivos. Como benefícios indiretos e coletivos, a pesquisa contribui para o avanço científico de sistemas gratuitos e reprodutíveis de auxílio ao diagnóstico precoce do câncer de pulmão, com potencial futuro de otimizar a triagem radiológica, reduzir a sobrecarga visual médica e apoiar a detecção de lesões em estágios iniciais com maior probabilidade de cura.

#### Metodologia de Análise de Dados
> A avaliação quantitativa do modelo será realizada por meio de métricas consagradas na literatura de detecção de objetos: Precisão (Precision), Sensibilidade/Revocação (Recall), Precisão Média Média com limiar IoU 0,50 (mAP@50), mAP@50-95, curvas de perda (Box Loss e Classification Loss) e Matriz de Confusão, além da verificação da taxa de falsos positivos em tomografias de pulmão saudável em diferentes limiares de confiança.

#### Desfecho Primário
> Obtenção de modelo detector YOLOv8s validado sem vazamento intra-paciente, capaz de localizar nódulos pulmonares com equilíbrio clinicamente consistente entre Precisão, Recall e controle de falsos positivos.

---

### Etapa 5 — Dispensa do TCLE (Marcar "Sim" para Solicitação de Dispensa do TCLE)
* **Justificativa para copiar no campo da Plataforma Brasil:**
> Solicita-se a dispensa da aplicação do Termo de Consentimento Livre e Esclarecido (TCLE), em conformidade com a Resolução CNS nº 466/2012 (item IV.8) e a Resolução CNS nº 510/2016 (art. 1º, parágrafo único), visto que a pesquisa utiliza exclusivamente dados secundários de imagens médicas provenientes de três repositórios públicos internacionais de acesso livre (LIDC-IDRI, IQ-OTH/NCCD e Chest CT-Scan Images), cujos registros foram previamente coletados e integralmente anonimizados pelos consórcios responsáveis. Não há qualquer contato direto ou indireto com participantes humanos, tampouco acesso a nomes, prontuários ou dados identificáveis, tornando inviável e desnecessária a obtenção de consentimento individual. Adicionalmente, os testes do protótipo de software desenvolvido no TCC empregarão apenas dados fictícios/simulados.
