"""
API REST para Inferência, Benchmark e Avaliação OOD no MNIST.
Substitui o app.py (Streamlit) por serviços desacoplados em FastAPI.
"""
import io
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, List, Any
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field

# Reutilização direta dos módulos existentes
from src.data_loader import load_mnist_data, split_stratified_data
from src.preprocessing import normalize_pixels, preprocess_custom_image
from src.models import get_configured_models, train_and_evaluate_all, build_comparison_dataframe
from src.ood_stress import run_class_masking_experiment

# -------------------------------------------------------------
# Armazenamento em Memória (Estado do Ciclo de Vida da API)
# -------------------------------------------------------------
ml_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carrega dados, treina/aquece os modelos e armazena em memória no startup da API,
    eliminando re-treinamentos desnecessários durante as requisições.
    """
    print("Inicializando pipeline de Machine Learning...")
    X_raw, y_raw = load_mnist_data()
    
    # Subamostragem controlada para inicialização rápida
    SAMPLE_SIZE = 12000
    np.random.seed(42)
    indices = np.random.choice(len(X_raw), SAMPLE_SIZE, replace=False)
    X_sample, y_sample = X_raw[indices], y_raw[indices]
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified_data(X_sample, y_sample)
    
    # Normalização
    X_train_norm = normalize_pixels(X_train)
    X_test_norm = normalize_pixels(X_test)
    
    # Treinamento e avaliação dos modelos configurados
    models = get_configured_models()
    results = train_and_evaluate_all(models, X_train_norm, y_train, X_test_norm, y_test)
    
    # Experimento OOD (Classes 4 e 7 mascaradas)
    ood_results = run_class_masking_experiment(
        X_train_norm, y_train, X_test_norm, y_test, masked_classes=(4, 7)
    )
    
    ml_state["models"] = models
    ml_state["results"] = results
    ml_state["ood_results"] = ood_results
    ml_state["best_model"] = results["Rede Neural (MLP Profunda)"]["model"]
    
    print("API pronta para inferência.")
    yield
    ml_state.clear()


app = FastAPI(
    title="MNIST Predictive AI Studio - REST API",
    version="2.0.0",
    description="Microsserviço de classificação de dígitos manuscritos, benchmark e análise de incerteza (OOD).",
    lifespan=lifespan
)

# -------------------------------------------------------------
# Schemas Pydantic (Validação de Entrada e Saída)
# -------------------------------------------------------------
class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: Dict[str, float]
    processed_shape: List[int] = [28, 28]

class VectorInput(BaseModel):
    pixels: List[float] = Field(..., description="Vetor com 784 floats normalizados entre 0.0 e 1.0")

class BenchmarkResponse(BaseModel):
    benchmark: List[Dict[str, Any]]

class OODResponse(BaseModel):
    masked_classes: List[int]
    mean_confidence: float
    median_confidence: float
    misclassification_distribution: Dict[str, int]

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/health", tags=["Status"])
def health_check():
    """Verifica a integridade do serviço e modelos em memória."""
    return {
        "status": "online",
        "models_loaded": list(ml_state.get("results", {}).keys())
    }


@app.get("/benchmark", response_model=BenchmarkResponse, tags=["Avaliação"])
def get_benchmark():
    """Retorna métricas comparativas consolidadas (Acurácia, Precisão, Recall, F1 e Tempos)."""
    if "results" not in ml_state:
        raise HTTPException(status_code=503, detail="Modelos ainda não inicializados.")
    
    df_comp = build_comparison_dataframe(ml_state["results"])
    return {"benchmark": df_comp.to_dict(orient="records")}


@app.get("/ood-evaluation", response_model=OODResponse, tags=["Avaliação"])
def get_ood_metrics():
    """Retorna métricas de estresse OOD e falsa certeza sobre dígitos 4 e 7."""
    if "ood_results" not in ml_state:
        raise HTTPException(status_code=503, detail="Dados OOD não disponíveis.")
    
    ood = ml_state["ood_results"]
    return {
        "masked_classes": list(ood["masked_classes"]),
        "mean_confidence": round(ood["mean_confidence"], 4),
        "median_confidence": round(ood["median_confidence"], 4),
        "misclassification_distribution": {str(k): int(v) for k, v in ood["pred_distribution"].items()}
    }


@app.post("/predict/image", response_model=PredictionResponse, tags=["Inferência"])
async def predict_from_image(file: UploadFile = File(...)):
    """
    Recebe imagem manuscrita (PNG/JPEG), aplica pipeline OpenCV/PIL
    (inversão de fundo, enquadramento e centro de massa) e retorna a classe.
    """
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. Envie arquivos PNG ou JPEG."
        )
    
    contents = await file.read()
    try:
        input_img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo de imagem corrompido.")
    
    # Pré-processamento unificado do projeto
    vector_784, _ = preprocess_custom_image(input_img)
    
    model = ml_state["best_model"]
    probs = model.predict_proba(vector_784)[0]
    pred_class = int(np.argmax(probs))
    
    return {
        "predicted_digit": pred_class,
        "confidence": float(probs[pred_class]),
        "probabilities": {str(i): float(probs[i]) for i in range(10)},
        "processed_shape": [28, 28]
    }


@app.post("/predict/vector", response_model=PredictionResponse, tags=["Inferência"])
def predict_from_vector(data: VectorInput):
    """Realiza a classificação a partir de um vetor já linearizado de 784 posições."""
    if len(data.pixels) != 784:
        raise HTTPException(status_code=400, detail="O vetor deve possuir exatamente 784 dimensões.")
    
    vec = np.array(data.pixels, dtype=np.float32).reshape(1, -1)
    model = ml_state["best_model"]
    probs = model.predict_proba(vec)[0]
    pred_class = int(np.argmax(probs))
    
    return {
        "predicted_digit": pred_class,
        "confidence": float(probs[pred_class]),
        "probabilities": {str(i): float(probs[i]) for i in range(10)},
        "processed_shape": [28, 28]
    }