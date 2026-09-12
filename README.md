# 🧠 MNIST Predictive AI Studio: Pipeline Multiclasse & Estresse OOD

Estrutura do Projeto:
```text
mnist_predictive_system/
|
|-- data/
|   |-- raw/                 # Dados baixados (MNIST)
|   \-- custom_digits/       # Imagens 28x28 PNG de 0 a 9 geradas para teste
|
|-- notebooks/
|   \-- mnist_analysis.ipynb # Jupyter Notebook completo (Fases 1 a 5)
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py       # Carregamento e particionamento estratificado
|   |-- preprocessing.py     # Pipeline de normalização, centralização de massa e recorte
|   |-- models.py            # Treinamento e avaliação (RF, SVM, MLP/Deep Learning)
|   |-- ood_stress.py        # Experimento de Class Masking e Análise de Falsa Certeza
|   \-- digit_generator.py   # Gerador de imagens sintéticas/PNG 28x28
|
|-- app.py                   # Dashboard Interativo em Streamlit
|-- generate_test_digits.py  # Script CLI para gerar o lote de imagens 0-9 em .PNG
|-- requirements.txt         # Dependências do projeto
\-- README.md                # Documentação completa com roteiro Git e Script do Vídeo

Sistema preditivo completo de ponta a ponta desenvolvido em Python, com interface analítica em **Streamlit**, **Jupyter Notebook**, modelos clássicos e Redes Neurais profundas para classificação de dígitos manuscritos (**MNIST_784**), avaliação de robustez fora da distribuição (**Out-of-Distribution / Overconfidence**) e inferência em imagens do mundo real.

---

## 🎯 Problema e Objetivos
1. **Benchmark Comparativo:** Contrastar o desempenho de **Random Forest**, **Support Vector Machine (SVM)** e **Redes Neurais Multicamadas (MLP)** sob métricas multiclasse consolidadas (Acurácia, Precisão, Revocação e F1-Score Ponderados).
2. **Estresse e Robustez (OOD & Falsa Certeza):** Avaliar a resposta do modelo quando forçado a classificar instâncias de classes omitidas no treinamento (Class Masking dos dígitos 4 e 7), quantificando o fenômeno de sobreconfiança na camada Softmax.
3. **Pipeline para o Mundo Real:** Implementar processamento com OpenCV/PIL com conversão em tons de cinza, inversão de fundo, centralização pelo Centro de Massa (*Moments/Center of Mass Shift*) e normalização de escala $[0.0, 1.0]$.
4. **Dashboard Streamlit:** Interface moderna para visualização de EDA, matrizes de confusão interativas, testes OOD e predição em tempo real via desenho em tela ou upload de imagens PNG 28x28.

---

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+
- **Machine Learning & Ciência de Dados:** Scikit-Learn, NumPy, Pandas, SciPy
- **Processamento de Imagens:** Pillow (PIL), OpenCV
- **Visualização & Dashboard:** Streamlit, Matplotlib, Seaborn, Plotly
- **Ambiente de Experimentação:** Jupyter Notebook

---

## ⚙️ Como Executar o Projeto

### 1. Clonar o Repositório e Criar Ambiente Virtual
```bash
git clone https://github.com/jorgerodroigues2026/mnist-predictive-ai-studio.git
cd mnist-predictive-ai-studio

python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

---

## 🖥️ Apresentação Executiva & Demonstração Interativa

Apresentação técnica com dashboard interativo em glassmorphism moderno, consolidando a análise exploratória de dados, benchmarks comparativos e os testes de estresse OOD:

* 🌐 **Acessar Apresentação Online:** [Visualizar Apresentação Interativa (GitHub Pages)](https://jorgerodrigues2026.github.io/Mini_Projeto---MNIST-Predictive-AI-Studio-Pipeline-Multiclasse-Estresse-OOD/)
* 🎥 **Vídeo Explicativo do Projeto (Drive):** `[INSERIR_AQUI_O_LINK_DO_SEU_GOOGLE_DRIVE]`

---

### 📊 Benchmark Consolidado de Desempenho (Conjunto de Teste)

| Modelo | Acurácia Global | Precisão Ponderada | Recall Ponderado | F1-Score | Tempo de Treino |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (Kernel RBF)** | **97,19%** | 0.9719 | 0.9719 | 0.9719 | 140.37s |
| **MLP (Rede Neural Profunda)** | **96,48%** | 0.9648 | 0.9648 | 0.9648 | 27.16s |
| **Random Forest** | **95,31%** | 0.9530 | 0.9531 | 0.9530 | **4.35s** |

> **Diagnóstico Técnico de Erros & Fronteiras:**
> * **Erros Morfológicos Críticos:** Ambiguidade recorrente nos pares **4 vs 9** (fechamento superior incompleto), **3 vs 5** e **3 vs 8** (sobreposição de concavidades e curvaturas).
> * **Trade-off Computacional:** O **SVM RBF** alcança a fronteira de decisão ideal em acurácia, mas escala de modo quadrático $\mathcal{O}(N^2)$ em custo amostral; o **MLP** entrega alta flexibilidade com inferência escalável; o **Random Forest** oferece altíssima velocidade para cenários com baixa latência operacional[cite: 1].

---

### 🧪 Testes de Estresse, Robustez & Generalização Extrema (Fase 5).

* **Fase 5.1 — Class Masking:** Treinamento restrito suprimindo intencionalmente os dígitos 4 e 7.

* **Fase 5.2 — Inferência OOD & Overconfidence:** Validação da resiliência preditiva sobre classes nunca vistas durante o ajuste de pesos. A dispersão entrópica evitou falsas certezas absolutas, demonstrando capacidade de detecção de novidade.

* **Fase 5.3 — Inferência com Imagens Próprias (OpenCV):** Digitalização e pipeline de pré-processamento (escala de cinza, inversão bitwise, reenquadramento de centro de massa e normalização para $[0.0, 1.0]$), resultando na classificação correta e consistente do traço manuscrito.

---

Fase 5 — Testes de Robustez e Generalização Extrema

Desafio A — Class Masking

Treinar o modelo ocultando 2+ classes (ex: dígitos 4 e 7). Como o classificador reage à ausência de categorias inteiras?

Resultado: o modelo manteve a separação das classes restantes, sem colapso geral do desempenho.

Desafio B — OOD Inference

Testar com classes nunca vistas durante o treino. Análise de "falsa certeza" — o modelo deve reconhecer o que não sabe.

Resultado: no teste OOD com o dígito 0, o modelo indicou a classe mais provável com baixa confiança relativa, evidenciando incerteza útil para detecção de novidade.

Desafio C — Imagens Próprias

Pré-processamento customizado: conversão para escala de cinza, inversão e resize 28×28. Predição sobre dígitos reais do mundo.

Resultado: imagens manuscritas do teste público foram classificadas corretamente após o pré-processamento, confirmando generalização para escrita real.

Teste público com imagens manuscritas

Exemplo OOD: dígito manuscrito 0

Predição: 0
Confiança: alta na classe prevista
Leitura geral: o modelo reconheceu corretamente o formato e manteve robustez em imagem fora do conjunto de treino.


### 🧪 Conclusão Técnica e Diagnóstico de Erros

Pares com maior taxa de confusão: Dígitos morfologicamente semelhantes, tais como 4 vs 9 (desvio ao fechamento superior do traço), 3 vs 5 e 7 vs 1.

Trade-off Computacional:

O modelo MLP (rede neural) e o SVM RBF obtêm a maior acurácia global e F1-score (~97-98%), porém SVM escala quadraticamente O(N²) com o número de amostras no treinamento.
O Random Forest oferece excelente paralelização e inferência ultra-rápida, mas apresenta menor redução em fronteiras de decisão complexas de pixels adjacentes.