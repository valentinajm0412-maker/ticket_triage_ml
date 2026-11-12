"""
Hoja CRISP-ML Fase 4: Evaluación del Modelo || CRISP-ML Phase 4 page: Model Evaluation
Lee ml_models/evaluation_report.json y las revisiones de expertos guardadas en la BD.
"""
from evaluation import EVALUATION_REPORT_PATH, load_evaluation
from model_comparison import load_comparison

VERDICTS = {
    'modelo': 'Error del modelo (la etiqueta real es correcta)',
    'etiqueta': 'Error de la etiqueta (el modelo tiene razón)',
    'ambiguo': 'Ambiguo (ambas prioridades son defendibles)',
}
PENDING_TO_SHOW = 10

# Pregunta de negocio -> modelo del curso (marco teórico)
COURSE_QUESTIONS = [
    {'pregunta': '¿Qué prioridad tiene el ticket?', 'modelo': 'Logistic Regression', 'tipo': 'Clasificación supervisada',
     'como': 'Calcula la probabilidad de cada prioridad a partir de las palabras del ticket.'},
    {'pregunta': '¿Qué prioridad tiene el ticket? (por tickets parecidos)', 'modelo': 'KNN', 'tipo': 'Clasificación supervisada',
     'como': 'Busca los K tickets históricos más parecidos y toma la prioridad más común entre ellos.'},
    {'pregunta': '¿Qué tan grave es? (puntaje Bajo=0 … Crítico=3)', 'modelo': 'Linear Regression', 'tipo': 'Regresión supervisada',
     'como': 'Predice un número de gravedad; por encima del corte es Crítico y el resto se redondea.'},
    {'pregunta': '¿Qué tipos de tickets llegan?', 'modelo': 'K-Means', 'tipo': 'Agrupamiento no supervisado',
     'como': 'Agrupa tickets con textos parecidos sin mirar la prioridad.'},
    {'pregunta': '¿Qué hacer con el ticket?', 'modelo': 'Q-Learning', 'tipo': 'Aprendizaje por refuerzo',
     'como': 'Aprende qué acción (atender ya, cola normal, revisión humana) da más recompensa según la prioridad predicha y la confianza.'},
]


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

    cmpr = load_comparison()
    chosen = next((r for r in cmpr['ranking_prioridad'] if r.get('en_produccion')), None) if cmpr else None
    best = cmpr['ranking_prioridad'][0] if cmpr else None
    return {
        'ev': ev, 'path': EVALUATION_REPORT_PATH, 'cmp': cmpr, 'chosen': chosen, 'best': best,
        'questions': COURSE_QUESTIONS,
        'cm_labels': cm['etiquetas'], 'cm_rows': list(zip(cm['etiquetas'], cm['valores'])), 'max_cell': max_cell,
        'pending': pending[:PENDING_TO_SHOW], 'pending_total': len(pending),
        'reviews': reviews[:15], 'verdicts': VERDICTS, 'counts': counts,
        'total_reviewed': total_reviewed, 'errors_total': errors, 'adjusted_accuracy': adjusted,
        'priorities': ['Critico', 'Alto', 'Medio', 'Bajo'],
    }
