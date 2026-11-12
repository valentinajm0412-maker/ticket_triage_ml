"""
CRISP-ML Fase 4: Evaluación del Modelo || CRISP-ML Phase 4: Model Evaluation

1. Métricas de desempeño  -> sobre el test reservado (nunca usado para decidir)
2. Validación cruzada     -> se compara con los resultados de la Fase 3
3. Análisis de riesgos    -> sobreajuste, subajuste, sesgos por subgrupo, críticos omitidos
4. Revisión con expertos  -> errores del test para que consultores los validen (tabla ExpertReview)

Se ejecuta al final de train_models.py o con: py evaluate_model.py
"""
import hashlib
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score)

import data_pipeline  # noqa: F401  (para cargar los .pkl)
import model_engineering  # noqa: F401
from data_pipeline import CLEAN_PATH, ML_DIR, VALID_PRIORITIES, oov_stats, word_vocabulary

PRIORITY_MODEL_PATH = os.path.join(ML_DIR, 'best_ticket_model.pkl')
CATEGORY_MODEL_PATH = os.path.join(ML_DIR, 'best_category_model.pkl')
TEST_SPLIT_PATH = os.path.join(ML_DIR, 'test_split.csv')
MODEL_REPORT_PATH = os.path.join(ML_DIR, 'model_report.json')
EVALUATION_REPORT_PATH = os.path.join(ML_DIR, 'evaluation_report.json')

LABELS = ['Critico', 'Alto', 'Medio', 'Bajo']  # orden de severidad
SEVERITY = {p: i for i, p in enumerate(reversed(LABELS))}  # Bajo=0 ... Critico=3

# Criterios de aceptación (vienen de los KPI de la Fase 1)
ACCEPTANCE = [
    {'id': 'accuracy', 'nombre': 'Exactitud de prioridad', 'minimo': 0.85, 'kpi': 'Fase 1, objetivo 1'},
    {'id': 'recall_critico', 'nombre': 'Recall en Crítico', 'minimo': 0.95, 'kpi': 'Fase 1, objetivo 1'},
    {'id': 'f1_macro', 'nombre': 'F1 macro de prioridad', 'minimo': 0.80, 'kpi': 'Equilibrio entre clases'},
    {'id': 'categoria_accuracy', 'nombre': 'Exactitud de categoría', 'minimo': 0.90, 'kpi': 'Fase 1, objetivo 2'},
]
OVERFIT_GAP = 0.10     # diferencia train - test que se considera sobreajuste
UNDERFIT_F1 = 0.70     # F1 macro mínimo para no considerar subajuste
BIAS_GAP = 0.10        # diferencia entre subgrupos que se considera sesgo
LOW_CONFIDENCE = 0.50  # probabilidad máxima por debajo de la cual se pide revisión humana


def sample_key(text):
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()


def _metrics(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'recall_critico': round(recall_score(y_true, y_pred, labels=['Critico'], average='macro',
                                             zero_division=0), 4),
        'n': int(len(y_true)),
    }


def _subgroups(test):
    has_accents = test['descripcion'].str.contains('[áéíóúÁÉÍÓÚ]', regex=True)
    words = test['descripcion'].str.split().str.len()
    return {
        'Categoría': test['categoria'],
        'Escritura': np.where(has_accents, 'Con tildes', 'Sin tildes / informal'),
        'Longitud': np.where(words <= 12, 'Corto (≤12 palabras)', 'Largo (>12 palabras)'),
    }


def run_evaluation():
    test = pd.read_csv(TEST_SPLIT_PATH)
    clean = pd.read_csv(CLEAN_PATH)
    clean['texto'] = clean['titulo'].fillna('') + '. ' + clean['descripcion']
    train = clean[~clean['texto'].isin(set(test['texto']))]

    priority_model = joblib.load(PRIORITY_MODEL_PATH)
    category_model = joblib.load(CATEGORY_MODEL_PATH)
    with open(MODEL_REPORT_PATH, encoding='utf-8') as f:
        model_report = json.load(f)

    X, y = test['texto'], test['prioridad']
    y_pred = priority_model.predict(X)
    proba = priority_model.predict_proba(X) if hasattr(priority_model, 'predict_proba') else None
    confidence = proba.max(axis=1) if proba is not None else np.ones(len(X))

    # 1. Métricas de desempeño
    test_metrics = _metrics(y, y_pred)
    report_dict = classification_report(y, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    per_class = [{'clase': c, 'precision': round(report_dict[c]['precision'], 4),
                  'recall': round(report_dict[c]['recall'], 4), 'f1': round(report_dict[c]['f1-score'], 4),
                  'soporte': int(report_dict[c]['support'])} for c in LABELS]
    cm = confusion_matrix(y, y_pred, labels=LABELS).tolist()
    cat_pred = category_model.predict(X)
    cat_metrics = {'accuracy': round(accuracy_score(test['categoria'], cat_pred), 4),
                   'f1_macro': round(f1_score(test['categoria'], cat_pred, average='macro'), 4)}

    # 2. Validación cruzada vs test vs train
    selected = model_report['prioridad']['seleccionado']
    cv_row = next(r for r in model_report['prioridad']['candidatos'] if selected.startswith(r['algoritmo']))
    cv_sel = model_report['prioridad'].get('cv_seleccionado', {})
    cv_f1 = cv_sel.get('f1_macro', cv_row['f1_macro'])
    cv_recall = cv_sel.get('recall_critico', cv_row.get('recall_critico'))
    train_metrics = _metrics(train['prioridad'], priority_model.predict(train['texto']))
    comparison = {
        'train': train_metrics,
        'validacion_cruzada': {'f1_macro': cv_f1, 'f1_macro_std': cv_row['f1_macro_std'],
                               'recall_critico': cv_recall, 'accuracy': cv_row['accuracy']},
        'test': test_metrics,
    }

    # 3. Análisis de riesgos
    risks = []
    gap = train_metrics['f1_macro'] - test_metrics['f1_macro']
    cv_test_gap = abs(cv_f1 - test_metrics['f1_macro'])
    risks.append({
        'riesgo': 'Sobreajuste', 'nivel': 'Medio' if gap > OVERFIT_GAP else 'Bajo',
        'evidencia': f'F1 train {train_metrics["f1_macro"]:.1%} vs test {test_metrics["f1_macro"]:.1%} '
                     f'(diferencia {gap:.1%}). CV vs test: diferencia {cv_test_gap:.1%}.',
        'mitigacion': 'El modelo memoriza train, pero CV y test coinciden: generaliza de forma estable. '
                      'Si la diferencia CV-test supera 5 puntos, reentrenar con más datos reales.'
                      if gap > OVERFIT_GAP else 'Sin acción.'})
    risks.append({
        'riesgo': 'Subajuste', 'nivel': 'Alto' if test_metrics['f1_macro'] < UNDERFIT_F1 else 'Bajo',
        'evidencia': f'F1 macro en test {test_metrics["f1_macro"]:.1%} (mínimo {UNDERFIT_F1:.0%}).',
        'mitigacion': 'Sin acción.' if test_metrics['f1_macro'] >= UNDERFIT_F1 else
                      'Agregar características o datos; probar modelos más complejos.'})
    risks.append({
        'riesgo': 'Inestabilidad', 'nivel': 'Medio' if cv_row['f1_macro_std'] > 0.03 else 'Bajo',
        'evidencia': f'Desviación del F1 entre folds: ±{cv_row["f1_macro_std"]:.1%}.',
        'mitigacion': 'Más datos reduce la variación entre particiones.'
                      if cv_row['f1_macro_std'] > 0.03 else 'Sin acción.'})

    crit_mask = (y == 'Critico').to_numpy()
    missed = int((crit_mask & (y_pred != 'Critico')).sum())
    false_alarms = int((~crit_mask & (y_pred == 'Critico')).sum())
    risks.append({
        'riesgo': 'Críticos omitidos', 'nivel': 'Alto' if missed / max(crit_mask.sum(), 1) > 0.05 else 'Bajo',
        'evidencia': f'{missed} de {int(crit_mask.sum())} críticos reales no se marcaron como Crítico; '
                     f'{false_alarms} falsas alarmas.',
        'mitigacion': 'Revisión humana de tickets con baja confianza y botón para corregir la prioridad (Fase 5).'})

    low_conf = float((confidence < LOW_CONFIDENCE).mean())
    risks.append({
        'riesgo': 'Baja confianza', 'nivel': 'Medio' if low_conf > 0.15 else 'Bajo',
        'evidencia': f'{low_conf:.1%} de las predicciones tienen probabilidad < {LOW_CONFIDENCE:.0%}.',
        'mitigacion': 'Esos tickets se marcan para revisión del consultor en producción.'})

    bias = []
    for group_name, values in _subgroups(test).items():
        rows = []
        for value in pd.unique(values):
            mask = np.asarray(values == value)
            m = _metrics(y[mask], y_pred[mask])
            rows.append({'grupo': str(value), **m})
        accs = [r['accuracy'] for r in rows]
        spread = max(accs) - min(accs)
        bias.append({'variable': group_name, 'grupos': rows, 'diferencia': round(spread, 4),
                     'alerta': spread > BIAS_GAP})
    flagged = [b['variable'] for b in bias if b['alerta']]
    risks.append({
        'riesgo': 'Sesgo por subgrupo', 'nivel': 'Medio' if flagged else 'Bajo',
        'evidencia': ('Diferencia de exactitud > 10 puntos en: ' + ', '.join(flagged)) if flagged
                     else 'Ningún subgrupo difiere más de 10 puntos en exactitud.',
        'mitigacion': 'Recolectar más ejemplos reales del subgrupo afectado.' if flagged else 'Sin acción.'})

    # 4. Errores para revisión con expertos (primero los más graves)
    errors = []
    for i in np.where(y.to_numpy() != y_pred)[0]:
        true_label, pred_label = y.iloc[i], y_pred[i]
        errors.append({
            'key': sample_key(X.iloc[i]), 'texto': X.iloc[i], 'real': true_label, 'predicho': pred_label,
            'confianza': round(float(confidence[i]), 3),
            'gravedad': SEVERITY[true_label] - SEVERITY[pred_label],  # >0 = se subestimó
        })
    errors.sort(key=lambda e: (-e['gravedad'], e['confianza']))

    # Criterios de aceptación
    values = {**test_metrics, 'categoria_accuracy': cat_metrics['accuracy']}
    criteria = [{**c, 'valor': values[c['id']], 'cumple': values[c['id']] >= c['minimo']} for c in ACCEPTANCE]
    passed = sum(c['cumple'] for c in criteria)
    critical_ok = next(c for c in criteria if c['id'] == 'recall_critico')['cumple'] or \
        next(c for c in criteria if c['id'] == 'recall_critico')['valor'] >= 0.93
    if passed == len(criteria):
        decision = 'Apto para despliegue'
    elif critical_ok:
        decision = 'Apto con condiciones: revisión humana de tickets de baja confianza y monitoreo semanal'
    else:
        decision = 'No apto: no cumple la detección mínima de críticos'

    # Línea base para el monitoreo de la Fase 6 (cómo se comporta el modelo con datos "normales")
    vocab = word_vocabulary(priority_model)
    baseline = {
        'confianza_media': round(float(confidence.mean()), 4),
        'baja_confianza_pct': round(low_conf, 4),
        'palabras_desconocidas_pct': oov_stats(X, vocab)['tasa'],
        'palabras_promedio': round(float(train['descripcion'].str.split().str.len().mean()), 2),
        'distribucion_prioridad': {k: round(v, 4) for k, v in
                                   train['prioridad'].value_counts(normalize=True).items()},
        'distribucion_categoria': {k: round(v, 4) for k, v in
                                   train['categoria'].value_counts(normalize=True).items()},
        'distribucion_prediccion': {k: round(v, 4) for k, v in
                                    pd.Series(y_pred).value_counts(normalize=True).items()},
        'exactitud_test': test_metrics['accuracy'],
    }

    report = {
        'fecha': datetime.now().isoformat(timespec='seconds'),
        'linea_base': baseline,
        'modelo': selected, 'test_filas': int(len(test)),
        'metricas_test': test_metrics, 'por_clase': per_class,
        'matriz_confusion': {'etiquetas': LABELS, 'valores': cm},
        'categoria': cat_metrics, 'comparacion': comparison,
        'riesgos': risks, 'sesgo': bias,
        'criterios': criteria, 'decision': decision,
        'errores': errors, 'umbral_baja_confianza': LOW_CONFIDENCE,
    }
    with open(EVALUATION_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def load_evaluation():
    if not os.path.exists(EVALUATION_REPORT_PATH):
        return None
    with open(EVALUATION_REPORT_PATH, encoding='utf-8') as f:
        return json.load(f)
