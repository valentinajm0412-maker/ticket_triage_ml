"""
Hoja CRISP-ML Fase 3: Ingeniería del Modelo || CRISP-ML Phase 3 page: Model Engineering
Lee ml_models/model_report.json (lo genera train_models.py).
"""
import json
import os

from data_pipeline import ML_DIR

MODEL_REPORT_PATH = os.path.join(ML_DIR, 'model_report.json')

ALGORITHM_NOTES = {
    'Naive Bayes': 'Probabilístico y muy rápido; buena línea base para texto.',
    'Logistic Regression': 'Lineal, interpretable y entrega probabilidades.',
    'Linear SVM': 'Lineal de márgenes; fuerte en texto, pero sin probabilidades.',
    'Random Forest': 'Conjunto de árboles; capta combinaciones (ej. "caído" + "toda la empresa").',
    'Linear Regression (ordinal)': 'Trata la prioridad como un orden (Bajo < Medio < Alto < Crítico) y predice un puntaje 0-3.',
}


def build_context():
    if not os.path.exists(MODEL_REPORT_PATH):
        return {'mr': None, 'path': MODEL_REPORT_PATH}
    with open(MODEL_REPORT_PATH, encoding='utf-8') as f:
        mr = json.load(f)
    return {'mr': mr, 'path': MODEL_REPORT_PATH, 'notes': ALGORITHM_NOTES}
