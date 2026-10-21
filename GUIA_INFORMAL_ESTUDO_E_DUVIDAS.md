# Guia Rápido e Informal: Suas Dúvidas Respondidas + Como Rodar Tudo Sem o PC Local

Este documento é o seu **resumo de bolso** (escrito de forma direta e sem enrolação acadêmica) para você ler no celular ou em qualquer lugar quando estiver fora de casa, revisar os conceitos antes de falar com orientador/banca e ter o código na mão sem depender do seu computador pessoal.

---

## 1. Por que tiramos tudo do PC local e deixamos 100% Remoto?

Você vai sair e ficar sem acesso ao seu PC local, mas na prática **deixar tudo 100% remoto foi a melhor decisão técnica do projeto** por três motivos:

1. **O PC local só rodava em CPU (sem GPU CUDA):**
   - Quando testamos rodar no seu computador, o PyTorch identificou `CUDA available: False` (estava rodando apenas no processador comum). Treinar 50 épocas com mais de 16.000 imagens em CPU levaria dias e travaria a máquina.
2. **Os bancos de dados pesam centenas de megas / gigas:**
   - Baixar 16.000 fotos no seu computador ou no seu Google Drive comeria espaço à toa.
   - No **Google Colab**, o comando `kagglehub.dataset_download(...)` baixa as fotos direto no computador emprestado do Google (na nuvem). Não gasta 1 byte do seu PC nem do seu Google Drive!
3. **Mobilidade total (GitHub + Colab):**
   - Agora todos os seus códigos, gráficos, relatórios e o texto em ABNT estão salvos no seu GitHub (`lealtais/TRAINING_IA_NODULES_LDC`).
   - Você consegue abrir tanto o GitHub quanto o Google Colab até pelo navegador do celular ou de qualquer outro computador na rua.

---

## 2. A Escolha dos Bancos de Dados (Por que usamos cada um?)

Se alguém te perguntar *"Por que você escolheu esses bancos de dados?"*, a resposta na ponta da língua é essa:

1. **DeepLesion (Descartado):**
   - Foi o primeiro teste, mas ele mistura lesão de tudo quanto é órgão (fígado, rim, pâncreas, abdômen e pulmão).
   - Como o seu TCC é focado **só em nódulo de pulmão**, o modelo ficou confuso e tirou nota baixa (`mAP50 = 0.329`). Por isso descartamos.
2. **LIDC-IDRI / `zhangweiled` (Escolhido para ensinar o que É nódulo):**
   - É o banco padrão-ouro mundial de tomografia de pulmão. Tem **875 pacientes** e **15.486 fatias (fotos `.png`) com nódulo**.
   - **O grande diferencial dele:** 4 médicos radiologistas diferentes desenharam a máscara de cada nódulo. A gente juntou as 4 marcações (se pelo menos 1 médico marcou, entrou na caixa) para não depender da opinião de um médico só.
3. **IQ-OTH/NCCD + Chest CT-Scan Images (Escolhidos para ensinar o que é PULMÃO SAUDÁVEL):**
   - **O problema do `zhangweiled`:** 100% das fotos dele têm nódulo! Não tem nenhuma pasta de paciente saudável lá dentro (aquelas 62 fatias vazias eram só erro de recorte na borda do nódulo).
   - Se você treina a IA **só** mostrando foto com nódulo, ela fica viciada e acha que **todo pulmão do mundo é obrigado a ter um nódulo**, gerando falso positivo em vaso sanguíneo e osso.
   - Por isso juntamos as pastas **`Normal`** desses dois bancos (`IQ-OTH/NCCD` com 416 fotos limpas + `Chest CT-Scan` com 215 fotos limpas = **~630 tomografias de pulmões 100% saudáveis**). E como já estão em PNG/JPG, não precisamos baixar aqueles 38 GB absurdos de DICOM e XML!

---

## 3. Todas as suas dúvidas explicadas de forma simples

### Dúvida A: "Por que separar por PACIENTE é melhor do que separar por FATIA?"
Pensa na tomografia como um **pão de forma fatiado**:
- Se o **Paciente João** tem um nódulo, esse mesmo nódulo aparece na **Fatia 10**, na **Fatia 11** e na **Fatia 12**.
- A Fatia 11 e a Fatia 12 do João são **praticamente a mesma foto** (só muda 1 milímetro).
- **O que acontecia antes (divisão por fatia):** O código sorteava as fotos soltas. Aí a Fatia 11 do João caía no **Treino** (estudo) e a Fatia 12 do João caía na **Validação** (a prova). Na hora da prova, a IA tirava uma nota super alta (`0.754`), mas só porque ela "colou" — ela já tinha visto a Fatia 11 daquele mesmo paciente no estudo!
- **O que o código faz agora (divisão por paciente):** O código sorteia as **pastas dos pacientes** (`LIDC-IDRI-0001` a `0875`).
  - **743 pacientes** vão inteiros pro **Treino**.
  - **132 pacientes** vão inteiros pra **Validação**.
  - Assim, todas as fatias do João ficam só no treino, e todas as fatias da Maria ficam só na prova. Zero cola (*zero data leakage*)!

---

### Dúvida B: "Por que não fazia sentido rodar o código novo ANTES de colocar as fotos de pulmão saudável?"
Você teve toda razão quando questionou isso!
- Se você rodasse um treino novo **só** mudando a divisão para paciente (sem colocar as fotos de pulmão saudável), o modelo **não ia melhorar nada** — pelo contrário, ele ia continuar vendo 100% de fotos com nódulo, ia continuar dando falso positivo, e você ia perder horas treinando duas vezes à toa.
- O certo é o que fizemos agora: **juntar primeiro os pacientes separados + as 630 fotos de pulmão saudável no mesmo código** e rodar o treinamento **uma única vez**.

---

### Dúvida C: "São 15.000 fatias com nódulo contra 600 fotos sem nódulo. Isso não fica desproporcional?"
**Não fica desproporcional!** E o motivo é muito legal de entender:
1. Primeiro: cada fatia do LIDC-IDRI é **1 arquivo `.png`**. Ou seja, os 875 pacientes geram **15.486 fotos `.png`**.
2. Em uma foto que **tem nódulo**, o nódulo é só uma bolinha pequena (ocupa uns **2% da foto**). Os outros **98% daquela mesma foto já são pulmão saudável, costela, coração e fundo preto**!
3. Ou seja: em todas as 15.486 fotos, o YOLO já está vendo 98% de pulmão normal ao redor do nódulo.
4. O próprio manual oficial do **YOLOv8 (Ultralytics)** diz: *"Adicione apenas de 1% a 10% de fotos totalmente sem objeto (background) para zerar os falsos positivos"*.
5. **630 fotos de pulmão limpo perto de 15.486 fotos dá ~4%** — exatamente dentro da faixa perfeita que o YOLOv8 pede!

---

### Dúvida D: "Como funciona essa história das pastas? `train` e `val` ficam na mesma pasta? Como o YOLO sabe qual foto tem nódulo e qual é saudável?"

1. **`train` e `val` NUNCA ficam na mesma pasta!**
   - Existe a pasta `images/train` (só para estudo) e a pasta `images/val` (só para a prova). Elas são 100% separadas.
   - O que vai para a mesma pasta são **todas as fatias de um mesmo paciente** (ex: todas as 15 fotos do Paciente 1 vão juntas para `train`).

2. **Por que não existe uma pasta `com_nodulo` e outra pasta `sem_nodulo`?**
   - Porque o YOLO não separa quem é doente e quem é saudável por pasta!
   - No YOLO, todas as fotos de estudo ficam dentro de `images/train/`.
   - Quem conta pro YOLO se aquela foto tem nódulo ou se é um pulmão saudável é o **arquivo `.txt` que tem exatamente o mesmo nome da foto**, lá na pasta `labels/train/`:

| Tipo de Foto | Arquivo na pasta `images/train/` | Arquivo na pasta `labels/train/` | O que está escrito dentro do `.txt`? | O que o YOLO entende? |
| :--- | :--- | :--- | :--- | :--- |
| **Com Nódulo (LIDC)** | `paciente01_fatia02.png` | `paciente01_fatia02.txt` | `0 0.512 0.430 0.080 0.075` | *"Tem um nódulo (classe 0) exatamente nessa coordenada X e Y!"* |
| **Pulmão Saudável** | `normal_saudavel_0001.png` | `normal_saudavel_0001.txt` | *(Arquivo 100% em branco / vazio)* | *"Não tem nada escrito! Esse pulmão é limpo, NÃO devo marcar nenhuma caixa aqui!"* |

---

### Dúvida E: "Pode misturar dois bancos diferentes sem dar problema?"
**Pode sim**, desde que a gente deixe as fotos com a mesma "cara" (mesmo padrão de cor).
- Por isso, no código, antes de salvar as fotos de pulmão saudável na pasta do YOLO, nós passamos o comando `im.convert("L")` (que transforma qualquer foto em **escala de cinza puro de 1 canal** e salva em `.png`, exatamente igual às fotos do LIDC-IDRI). Assim o modelo não diferencia de qual banco a foto veio!

---

## 4. Código Pronto para Rodar no Google Colab (De Qualquer Lugar)

Quando quiser rodar ou mostrar rodando no Google Colab (mesmo fora do seu PC), é só abrir o Colab, ativar a **GPU T4** e colar o código do arquivo [`train_lidc_por_paciente.py`](train_lidc_por_paciente.py) que já está salvo aqui no seu repositório!
