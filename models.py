"""
Módulo de construção, treinamento e avaliação de modelos de Machine Learning e Redes Neurais.
"""
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def get_configured_models():
    """
    Retorna dicionário com 3 modelos com hiperparâmetros ajustados:
    1. Random Forest: n_estimators=150, max_depth=20, min_samples_split=4
    2. Support Vector Machine (SVC): C=10.0, kernel='rbf', gamma='scale'
    3. Multi-Layer Perceptron (MLP): hidden_layer_sizes=(256, 128), activation='relu',
       alpha=0.0001 (regularização L2), learning_rate_init=0.001
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "Support Vector Machine (RBF)": SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        ),
        "Rede Neural (MLP Profunda)": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=30,
            early_stopping=True,
            random_state=42
        )
    }
    return models


def train_and_evaluate_all(models, X_train, y_train, X_test, y_test):
    """
    Treina e avalia todos os modelos, computando tempos e métricas ponderadas.
    """
    results = {}
    
    for name, model in models.items():
        print(f"--- Treinando modelo: {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0_inf = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - t0_inf
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "model": model,
            "train_time_sec": train_time,
            "inference_time_sec": inference_time,
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "f1_weighted": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred
        }
        print(f"Finalizado: {name} | Acurácia: {acc:.4f} | F1: {f1:.4f} | Tempo Treino: {train_time:.2f}s")
        
    return results


def build_comparison_dataframe(results):
    """
    Gera DataFrame comparativo consolidado.
    """
    data = []
    for name, res in results.items():
        data.append({
            "Modelo": name,
            "Acurácia Global": f"{res['accuracy'] * 100:.2f}%",
            "Precisão Ponderada": f"{res['precision_weighted'] * 100:.2f}%",
            "Revocação Ponderada": f"{res['recall_weighted'] * 100:.2f}%",
            "F1-Score Ponderado": f"{res['f1_weighted'] * 100:.2f}%",
            "Tempo de Treino (s)": f"{res['train_time_sec']:.2f} s",
            "Tempo de Inferência (s)": f"{res['inference_time_sec']:.2f} s"
        })
    return pd.DataFrame(data)