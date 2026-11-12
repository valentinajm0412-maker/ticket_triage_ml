"""
CRISP-ML Fase 6: Monitoreo y Mantenimiento || CRISP-ML Phase 6: Monitoring and Maintenance

1. Supervisión continua      -> weekly_series(), current_metrics()
2. Detección de degradación  -> drift_checks(): PSI de prioridades/categorías, palabras nuevas, longitud
3. Actualización del modelo  -> retraining_triggers() + model_history.json + retrain.py (con reversión)
4. Revisión de riesgos       -> production_risks(): críticos omitidos, falsas alarmas, KPI de negocio

Línea base: el comportamiento del modelo en el test de la Fase 4 (evaluation_report.json -> linea_base).
"""
import json
import math
import os
from datetime import datetime, timedelta
from statistics import median

from data_pipeline import ML_DIR, oov_stats, word_vocabulary
from evaluation import load_evaluation

HISTORY_PATH = os.path.join(ML_DIR, 'model_history.json')

WINDOW_DAYS = 30            # ventana que se compara contra la línea base
WEEKS_SHOWN = 8
MIN_TICKETS = 20            # mínimo de tickets en la ventana para sacar conclusiones
MIN_REVIEWED = 20           # mínimo de revisiones para medir el acuerdo
AGREEMENT_DROP = 0.10       # caída del acuerdo frente a la exactitud de test
CONFIDENCE_DROP = 0.10      # caída de la confianza media
PSI_WARN, PSI_ALERT = 0.10, 0.25
RETRAIN_MIN_FEEDBACK = 50   # revisiones nuevas desde el último entrenamiento
MAX_MODEL_AGE_DAYS = 90
KPI_MAX_WORKLOAD_GAP = 2    # Fase 1, objetivo 2
KPI_CRITICAL_RESPONSE_MIN = 30  # Fase 1, objetivo 3

LEVELS = {'ok': 0, 'atencion': 1, 'alerta': 2}


def _psi(expected, actual, eps=0.001):
    """Population Stability Index: < 0.10 estable, 0.10-0.25 cambio moderado, > 0.25 cambio fuerte."""
    keys = set(expected) | set(actual)
    return round(sum((actual.get(k, 0) + eps - expected.get(k, 0) - eps)
                     * math.log((actual.get(k, 0) + eps) / (expected.get(k, 0) + eps)) for k in keys), 4)


def _distribution(values):
    values = [v for v in values if v]
    return {k: values.count(k) / len(values) for k in set(values)} if values else {}


def _level(alert, warn=False):
    return 'alerta' if alert else ('atencion' if warn else 'ok')


def _agreement(tickets):
    reviewed = [t for t in tickets if t.priority_reviewed_by and t.priority_predicted]
    if not reviewed:
        return None, 0
    return sum(t.priority == t.priority_predicted for t in reviewed) / len(reviewed), len(reviewed)


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. Supervisión continua
def weekly_series(tickets, now):
    start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    weeks = []
    for i in range(WEEKS_SHOWN - 1, -1, -1):
        ws, we = start - timedelta(weeks=i), start - timedelta(weeks=i - 1)
        wk = [t for t in tickets if t.created_at and ws <= t.created_at < we]
        conf = [t.priority_confidence for t in wk if t.priority_confidence is not None]
        agreement, n_rev = _agreement(wk)
        weeks.append({
            'semana': ws.strftime('%d %b'), 'tickets': len(wk), 'revisados': n_rev,
            'acuerdo': agreement, 'confianza': (sum(conf) / len(conf)) if conf else None,
            'baja_confianza': (sum(c < 0.5 for c in conf) / len(conf)) if conf else None,
            'criticos': sum(t.priority_predicted == 'Critico' for t in wk),
        })
    return weeks


def current_metrics(window, base):
    conf = [t.priority_confidence for t in window if t.priority_confidence is not None]
    agreement, n_rev = _agreement(window)
    conf_mean = (sum(conf) / len(conf)) if conf else None
    low = (sum(c < 0.5 for c in conf) / len(conf)) if conf else None
    enough = len(window) >= MIN_TICKETS
    rows = [
        {'indicador': 'Tickets clasificados', 'actual': len(window), 'base': None, 'formato': 'n',
         'estado': 'ok', 'nota': f'Últimos {WINDOW_DAYS} días'},
        {'indicador': 'Acuerdo modelo-consultor', 'actual': agreement, 'base': base.get('exactitud_test'),
         'formato': '%', 'nota': f'{n_rev} tickets revisados; la base es la exactitud en test',
         'estado': _level(n_rev >= MIN_REVIEWED and agreement is not None
                          and agreement < base.get('exactitud_test', 0) - AGREEMENT_DROP,
                          n_rev < MIN_REVIEWED)},
        {'indicador': 'Confianza media del modelo', 'actual': conf_mean, 'base': base.get('confianza_media'),
         'formato': '%', 'nota': 'Probabilidad que el modelo asigna a su respuesta',
         'estado': _level(enough and conf_mean is not None
                          and conf_mean < base.get('confianza_media', 0) - CONFIDENCE_DROP)},
        {'indicador': 'Tickets con baja confianza (⚠)', 'actual': low, 'base': base.get('baja_confianza_pct'),
         'formato': '%', 'nota': 'Van a revisión humana',
         'estado': _level(enough and low is not None and low > base.get('baja_confianza_pct', 0) + 0.10)},
        {'indicador': 'Tickets sin clasificar', 'actual': sum(1 for t in window if not t.priority_predicted),
         'base': 0, 'formato': 'n', 'nota': 'El modelo falló o no estaba cargado',
         'estado': _level(any(not t.priority_predicted for t in window))},
    ]
    return rows


# ---------------------------------------------------------------- 2. Detección de degradación
def drift_checks(window, base, model):
    enough = len(window) >= MIN_TICKETS
    checks = []

    psi_p = _psi(base.get('distribucion_prediccion', {}), _distribution([t.priority_predicted for t in window]))
    checks.append({'senal': 'Distribución de prioridades predichas', 'metrica': f'PSI {psi_p}',
                   'detalle': 'Compara cuántos Críticos/Altos/Medios/Bajos predice hoy vs. en test',
                   'estado': _level(enough and psi_p > PSI_ALERT, enough and psi_p > PSI_WARN)})

    psi_c = _psi(base.get('distribucion_categoria', {}), _distribution([t.category for t in window]))
    checks.append({'senal': 'Mezcla Incidente / Requerimiento', 'metrica': f'PSI {psi_c}',
                   'detalle': 'Cambio en el tipo de casos que llegan',
                   'estado': _level(enough and psi_c > PSI_ALERT, enough and psi_c > PSI_WARN)})

    top_words, oov_rate = [], None
    if model is not None and window:
        stats = oov_stats([f'{t.title}. {t.description}' for t in window], word_vocabulary(model))
        oov_rate, top_words = stats['tasa'], stats['top']
        base_oov = base.get('palabras_desconocidas_pct', 0)
        checks.append({'senal': 'Palabras que el modelo no conoce', 'metrica': f'{oov_rate:.1%} (base {base_oov:.1%})',
                       'detalle': 'Sistemas, módulos o formas de escribir nuevas',
                       'estado': _level(enough and oov_rate > max(2 * base_oov, base_oov + 0.10),
                                        enough and oov_rate > base_oov + 0.05)})

    lengths = [len(t.description.split()) for t in window if t.description]
    if lengths:
        avg = sum(lengths) / len(lengths)
        base_len = base.get('palabras_promedio', avg) or avg
        ratio = avg / base_len
        checks.append({'senal': 'Longitud de las descripciones', 'metrica': f'{avg:.1f} palabras (base {base_len:.1f})',
                       'detalle': 'Textos mucho más cortos o largos que los de entrenamiento',
                       'estado': _level(enough and (ratio < 0.5 or ratio > 2), enough and (ratio < 0.7 or ratio > 1.5))})
    return checks, top_words


# ---------------------------------------------------------------- 3. Actualización del modelo
def retraining_triggers(tickets, drift, metrics, history, model_report_date):
    trained_at = datetime.fromisoformat(model_report_date) if model_report_date else None
    new_feedback = sum(1 for t in tickets if t.priority_reviewed_at
                       and (trained_at is None or t.priority_reviewed_at > trained_at))
    age = (datetime.now() - trained_at).days if trained_at else None
    agreement_row = next(m for m in metrics if m['indicador'].startswith('Acuerdo'))
    triggers = [
        {'criterio': f'≥ {RETRAIN_MIN_FEEDBACK} revisiones nuevas desde el último entrenamiento',
         'valor': f'{new_feedback} revisiones', 'activo': new_feedback >= RETRAIN_MIN_FEEDBACK},
        {'criterio': 'Degradación detectada (alguna señal en alerta)',
         'valor': ', '.join(d['senal'] for d in drift if d['estado'] == 'alerta') or 'ninguna',
         'activo': any(d['estado'] == 'alerta' for d in drift)},
        {'criterio': f'Acuerdo modelo-consultor cae más de {AGREEMENT_DROP:.0%}',
         'valor': agreement_row['estado'], 'activo': agreement_row['estado'] == 'alerta'},
        {'criterio': f'Modelo con más de {MAX_MODEL_AGE_DAYS} días',
         'valor': f'{age} días' if age is not None else 'desconocido',
         'activo': age is not None and age > MAX_MODEL_AGE_DAYS},
    ]
    return triggers, new_feedback


# ---------------------------------------------------------------- 4. Revisión de riesgos
def production_risks(window, users, model_loaded, ev):
    # Línea base de la Fase 4: falsas alarmas y brecha entre categorías esperadas
    crit_row = next((c for c in ev.get('por_clase', []) if c['clase'] == 'Critico'), {})
    base_fa = 1 - crit_row.get('precision', 0.65)
    base_gap = next((b['diferencia'] for b in ev.get('sesgo', []) if b['variable'] == 'Categoría'), 0.15)

    reviewed = [t for t in window if t.priority_reviewed_by and t.priority_predicted]
    real_crit = [t for t in reviewed if t.priority == 'Critico']
    missed = [t for t in real_crit if t.priority_predicted != 'Critico']
    pred_crit = [t for t in reviewed if t.priority_predicted == 'Critico']
    false_alarms = [t for t in pred_crit if t.priority != 'Critico']

    risks = [{
        'riesgo': 'Críticos omitidos en producción',
        'evidencia': f'{len(missed)} de {len(real_crit)} críticos confirmados por consultores no fueron marcados Crítico',
        'estado': _level(bool(real_crit) and len(missed) / len(real_crit) > 0.05, bool(missed)),
        'accion': 'Revisar esos tickets en Evaluación y reentrenar con ellos.'}, {
        'riesgo': 'Falsas alarmas de Crítico',
        'evidencia': f'{len(false_alarms)} de {len(pred_crit)} tickets predichos Crítico fueron bajados por el consultor '
                     f'(esperado en test: {base_fa:.0%})',
        'estado': _level(False, len(pred_crit) >= MIN_REVIEWED and len(false_alarms) / len(pred_crit) > base_fa + 0.15),
        'accion': 'Si superan lo esperado en 15 puntos, subir el umbral de Crítico (Fase 3) tras reentrenar.'},
    ]

    for role in ['Consultor Tecnico', 'Consultor Funcional']:
        loads = [u.workload or 0 for u in users if u.role == role]
        if loads:
            gap = max(loads) - min(loads)
            risks.append({'riesgo': f'KPI carga balanceada ({role})',
                          'evidencia': f'Diferencia de carga entre consultores: {gap} tickets (meta ≤ {KPI_MAX_WORKLOAD_GAP})',
                          'estado': _level(gap > KPI_MAX_WORKLOAD_GAP), 'accion': 'Revisar asignación o agregar consultores.'})

    times = [(t.started_at - t.created_at).total_seconds() / 60 for t in window
             if t.priority == 'Critico' and t.started_at and t.created_at and t.started_at >= t.created_at]
    if times:
        med = median(times)
        risks.append({'riesgo': 'KPI primera respuesta a Críticos',
                      'evidencia': f'Mediana {med:.0f} min en {len(times)} críticos atendidos (meta < {KPI_CRITICAL_RESPONSE_MIN} min)',
                      'estado': _level(med > KPI_CRITICAL_RESPONSE_MIN), 'accion': 'Alertar a consultores cuando entra un Crítico.'})
    else:
        risks.append({'riesgo': 'KPI primera respuesta a Críticos', 'evidencia': 'Aún no hay críticos atendidos en la ventana',
                      'estado': 'ok', 'accion': 'Se mide desde "Tomar caso".'})

    by_cat = {}
    for cat in ['Incidente', 'Requerimiento']:
        agreement, n = _agreement([t for t in window if t.category == cat])
        if n >= MIN_REVIEWED:
            by_cat[cat] = agreement
    if len(by_cat) == 2:
        gap = abs(by_cat['Incidente'] - by_cat['Requerimiento'])
        risks.append({'riesgo': 'Sesgo por categoría en producción',
                      'evidencia': f"Acuerdo Incidente {by_cat['Incidente']:.0%} vs Requerimiento "
                                   f"{by_cat['Requerimiento']:.0%} (brecha {gap:.0%}; en test {base_gap:.0%})",
                      'estado': _level(False, gap > base_gap + 0.10),
                      'accion': 'Recolectar más ejemplos reales del grupo con menor acuerdo.'})
    else:
        risks.append({'riesgo': 'Sesgo por categoría en producción',
                      'evidencia': f'Se necesitan {MIN_REVIEWED} revisiones por categoría para medirlo',
                      'estado': 'ok', 'accion': 'Seguir confirmando prioridades en el Dashboard.'})

    risks.append({'riesgo': 'Disponibilidad del modelo', 'evidencia': 'Modelo cargado' if model_loaded else 'El modelo no está cargado',
                  'estado': _level(not model_loaded), 'accion': 'Ver protocolo de incidencias (Fase 5).'})
    return risks


def build_context(ticket_model, user_model, model, model_version):
    ev = load_evaluation() or {}
    base = ev.get('linea_base', {})
    now = datetime.now()
    tickets = ticket_model.query.all()
    window = [t for t in tickets if t.created_at and t.created_at >= now - timedelta(days=WINDOW_DAYS)]

    metrics = current_metrics(window, base)
    drift, top_words = drift_checks(window, base, model)
    history = load_history()
    model_date = None
    report_path = os.path.join(ML_DIR, 'model_report.json')
    if os.path.exists(report_path):
        with open(report_path, encoding='utf-8') as f:
            model_date = json.load(f).get('fecha')
    triggers, new_feedback = retraining_triggers(tickets, drift, metrics, history, model_date)
    risks = production_risks(window, user_model.query.all(), model is not None, ev)

    worst = max([LEVELS[x['estado']] for x in metrics + drift + risks] + [0])
    status = ['Estable', 'Atención', 'Acción requerida'][worst]
    retrain = any(t['activo'] for t in triggers)

    return {
        'has_baseline': bool(base), 'base': base, 'window_days': WINDOW_DAYS, 'window_n': len(window),
        'min_tickets': MIN_TICKETS, 'status': status, 'status_level': worst, 'retrain': retrain,
        'weeks': weekly_series(tickets, now), 'metrics': metrics, 'drift': drift, 'top_words': top_words,
        'triggers': triggers, 'new_feedback': new_feedback, 'history': list(reversed(history)),
        'risks': risks, 'model_version': model_version, 'psi_warn': PSI_WARN, 'psi_alert': PSI_ALERT,
    }
