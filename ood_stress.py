"""
Módulo de Testes de Estresse e Robustez (Out-of-Distribution / Falsa Certeza).
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def run_class_masking_experiment(X_train, y_train, X_test, y_test, masked_classes=(4, 7)):
    """
    Desafio A & B:
    1. Remove as classes especificadas (ex: 4 e 7) completamente da base de treino.
    2. Treina o modelo sem jamais ter visto essas classes.
    3. Avalia o modelo exclusivamente no subconjunto de teste OOD (composto apenas por 4 e 7).
    4. Analisa as probabilidades atribuídas e o fenômeno de Overconfidence.
    """
    # Filtra dados de treino (Classes Conhecidas)
    train_mask = ~np.isin(y_train, masked_classes)
    X_train_masked = X_train[train_mask]
    y_train_masked = y_train[train_mask]

    # Filtra dados de teste OOD (Classes Ocultadas)
    ood_test_mask = np.isin(y_test, masked_classes)
    X_test_ood = X_test[ood_test_mask]
    y_test_ood = y_test[ood_test_mask]

    # Modelo treinado no cenário restrito
    model_masked = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=25,
        random_state=42
    )
    model_masked.fit(X_train_masked, y_train_masked)

    # Inferência no conjunto OOD
    y_pred_ood = model_masked.predict(X_test_ood)
    probs_ood = model_masked.predict_proba(X_test_ood)
    max_confidence = np.max(probs_ood, axis=1)

    # Distribuição de atribuição de classes
    unique_preds, counts = np.unique(y_pred_ood, return_counts=True)
    pred_distribution = dict(zip(unique_preds, counts))

    return {
        "model_masked": model_masked,
        "masked_classes": masked_classes,
        "X_test_ood": X_test_ood,
        "y_test_ood": y_test_ood,
        "y_pred_ood": y_pred_ood,
        "probs_ood": probs_ood,
        "mean_confidence": float(np.mean(max_confidence)),
        "median_confidence": float(np.median(max_confidence)),
        "pred_distribution": pred_distribution
    }