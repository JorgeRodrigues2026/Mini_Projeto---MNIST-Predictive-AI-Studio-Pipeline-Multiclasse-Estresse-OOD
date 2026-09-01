"""
Módulo de carregamento e particionamento dos dados MNIST_784.
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist_data():
    """
    Baixa e carrega o dataset MNIST_784.
    Retorna X (matriz de pixels) e y (rótulos como inteiros de 0 a 9).
    """
    print("Baixando/Carregando dataset MNIST_784...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    print(f"Dataset carregado com sucesso! Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y


def split_stratified_data(X, y, train_size=0.70, val_size=0.10, test_size=0.20, random_state=42):
    """
    Realiza divisão estratificada em Treino, Validação e Teste (ex: 70% / 10% / 20%).
    Garante a proporcionalidade exata de cada dígito em todos os subconjuntos.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), "As proporções devem somar 1.0"
    
    # Primeira divisão: separa conjunto de teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Segunda divisão: separa treino e validação
    relative_val_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, stratify=y_train_val, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test