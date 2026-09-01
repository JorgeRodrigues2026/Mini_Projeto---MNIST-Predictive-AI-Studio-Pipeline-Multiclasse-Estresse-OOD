# 🧠 MNIST Predictive AI Studio: Pipeline Multiclasse & Estresse OOD

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
git clone https://github.com/SEU_USUARIO/mnist-predictive-ai-studio.git
cd mnist-predictive-ai-studio

python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate