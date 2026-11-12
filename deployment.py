"""
CRISP-ML Fase 5: Despliegue del Modelo || CRISP-ML Phase 5: Model Deployment

1. Implementación en producción       -> deployment_checklist()
2. Documentación y capacitación       -> README.md + guía por rol (TRAINING_GUIDE)
3. Protocolos (monitoreo, incidencias,
   retroalimentación)                 -> feedback_stats(), export_feedback(), INCIDENT_PROTOCOL
"""
import os
from datetime import datetime

import pandas as pd

import config
from data_pipeline import REAL_PATH, environment_info, load_report

TRAINING_GUIDE = [
    {'rol': 'Cliente', 'tareas': [
        'Crear ticket: título corto y descripción con qué falla, desde cuándo y a cuántos usuarios afecta.',
        'Elegir Incidente (algo dejó de funcionar) o Requerimiento (algo nuevo o un cambio).',
        'La prioridad la asigna el modelo; no hace falta escribir "urgente".']},
    {'rol': 'Consultor', 'tareas': [
        'Atender primero los tickets Críticos y los marcados con ⚠ (baja confianza del modelo).',
        'Confirmar o corregir la prioridad en el Dashboard: cada corrección entrena al próximo modelo.',
        'Tomar caso → Resolver, para que la carga de trabajo se libere.']},
    {'rol': 'Administrador', 'tareas': [
        'Revisar semanalmente la hoja de Despliegue: acuerdo modelo-consultor y tickets en revisión.',
        'Exportar la retroalimentación y reentrenar: py train_models.py.',
        'Revisar errores en la hoja de Evaluación con los consultores expertos.']},
]

INCIDENT_PROTOCOL = [
    {'situacion': 'El modelo no carga o falla al predecir',
     'senal': '/health muestra model_loaded = false; tickets con prioridad "Sin clasificar"',
     'accion': 'Los tickets quedan en revisión manual (la app no se cae). Revisar logs/predictions.log, '
               'reinstalar requirements.txt y reentrenar si cambió la versión de scikit-learn.'},
    {'situacion': 'Muchas correcciones de prioridad',
     'senal': 'Acuerdo modelo-consultor menor a 80% en esta hoja',
     'accion': 'Exportar retroalimentación, reentrenar y comparar en Evaluación antes de reemplazar el modelo.'},
    {'situacion': 'Un crítico real quedó con prioridad baja',
     'senal': 'Reporte del cliente o consultor',
     'accion': 'Corregir la prioridad de inmediato en el Dashboard y registrar el caso en la revisión de expertos.'},
    {'situacion': 'Demasiados tickets en revisión (⚠)',
     'senal': 'Más del 20% de los tickets nuevos con baja confianza',
     'accion': 'Indica datos distintos a los de entrenamiento (posible drift): reentrenar con tickets reales.'},
]


def deployment_checklist(app, admin_user=None, bcrypt=None, model_error=None, model_version=None):
    report = load_report() or {}
    trained = report.get('entorno', {})
    current = environment_info()
    versions_ok = all(trained.get(k) == current[k] for k in ('pandas', 'scikit-learn')) if trained else False

    default_admin_pw = bool(admin_user and bcrypt and bcrypt.check_password_hash(admin_user.password_hash, 'admin123'))
    try:
        import waitress  # noqa: F401
        has_waitress = True
    except ImportError:
        has_waitress = False

    items = [
        ('Clave secreta definida por variable de entorno', config.SECRET_KEY_FROM_ENV,
         'Definir $env:SECRET_KEY antes de arrancar.'),
        ('Modo debug apagado', not app.debug, 'No usar FLASK_DEBUG=1 en producción.'),
        ('Protección CSRF en formularios', app.config.get('WTF_CSRF_ENABLED', True), 'Flask-WTF activo.'),
        ('Modelo cargado', model_error is None, model_error or f'Versión {model_version}'),
        ('Versiones iguales a las del entrenamiento', versions_ok,
         f"Entrenado con scikit-learn {trained.get('scikit-learn', '?')}, ahora {current['scikit-learn']}"),
        ('Contraseña del admin cambiada', not default_admin_pw, 'El admin aún usa admin123: cambiarla en Gestionar Usuarios.'),
        ('Servidor de producción (waitress) instalado', has_waitress, 'py -m pip install -r requirements.txt'),
        ('Entorno marcado como producción', config.IS_PRODUCTION, '$env:APP_ENV = "production"'),
    ]
    return [{'item': i, 'ok': bool(ok), 'detalle': d} for i, ok, d in items]


def feedback_stats(ticket_model):
    tickets = ticket_model.query.all()
    predicted = [t for t in tickets if t.priority_predicted]
    reviewed = [t for t in predicted if t.priority_reviewed_by]
    agree = [t for t in reviewed if t.priority == t.priority_predicted]
    pending = [t for t in tickets if t.needs_review and not t.priority_reviewed_by]
    corrections = sorted([t for t in reviewed if t.priority != t.priority_predicted],
                         key=lambda t: t.priority_reviewed_at or datetime.min, reverse=True)
    confidences = [t.priority_confidence for t in predicted if t.priority_confidence is not None]
    return {
        'total': len(tickets), 'predichos': len(predicted), 'revisados': len(reviewed),
        'acuerdo': (len(agree) / len(reviewed)) if reviewed else None,
        'pendientes_revision': len(pending),
        'sin_clasificar': sum(1 for t in tickets if not t.priority),
        'confianza_promedio': (sum(confidences) / len(confidences)) if confidences else None,
        'correcciones': corrections[:10],
        'exportables': len(reviewed),
    }


def export_feedback(ticket_model):
    """Guarda los tickets revisados por consultores como fuente 'real' de la Fase 2."""
    rows = [{'titulo': t.title, 'descripcion': t.description, 'categoria': t.category,
             'prioridad': t.priority}
            for t in ticket_model.query.filter(ticket_model.priority_reviewed_by.isnot(None)).all()
            if t.priority]
    if not rows:
        return 0, REAL_PATH
    os.makedirs(os.path.dirname(REAL_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(REAL_PATH, index=False)
    return len(rows), REAL_PATH
