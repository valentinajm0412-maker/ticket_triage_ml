"""
Hoja CRISP-ML Fase 4: Evaluación del Modelo || CRISP-ML Phase 4 page: Model Evaluation
Lee ml_models/evaluation_report.json y las revisiones de expertos guardadas en la BD.
"""
from evaluation import EVALUATION_REPORT_PATH, load_evaluation

VERDICTS = {
    'modelo': 'Error del modelo (la etiqueta real es correcta)',
    'etiqueta': 'Error de la etiqueta (el modelo tiene razón)',
    'ambiguo': 'Ambiguo (ambas prioridades son defendibles)',
}
PENDING_TO_SHOW = 10


def build_context(review_model):
    ev = load_evaluation()
    if ev is None:
        return {'ev': None, 'path': EVALUATION_REPORT_PATH}

    reviews = review_model.query.order_by(review_model.created_at.desc()).all()
    reviewed_keys = {r.sample_key for r in reviews}
    pending = [e for e in ev['errores'] if e['key'] not in reviewed_keys]

    counts = {k: sum(1 for r in reviews if r.verdict == k) for k in VERDICTS}
    total_reviewed = len(reviews)
    n_test = ev['test_filas']
    errors = len(ev['errores'])
    # Exactitud ajustada: si el experto dice que la etiqueta estaba mal, el modelo acertó
    adjusted = (n_test - errors + counts['etiqueta']) / n_test if n_test else 0

    cm = ev['matriz_confusion']
    max_cell = max(max(row) for row in cm['valores']) or 1

    return {
        'ev': ev, 'path': EVALUATION_REPORT_PATH,
        'cm_labels': cm['etiquetas'], 'cm_rows': list(zip(cm['etiquetas'], cm['valores'])), 'max_cell': max_cell,
        'pending': pending[:PENDING_TO_SHOW], 'pending_total': len(pending),
        'reviews': reviews[:15], 'verdicts': VERDICTS, 'counts': counts,
        'total_reviewed': total_reviewed, 'errors_total': errors, 'adjusted_accuracy': adjusted,
        'priorities': ['Critico', 'Alto', 'Medio', 'Bajo'],
    }
