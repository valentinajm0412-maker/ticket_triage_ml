import hashlib
import logging
import os

import joblib
from models import db, User, Ticket

import data_pipeline  # noqa: F401  (needed to unpickle TextCleaner || necesario para cargar TextCleaner)
import model_engineering  # noqa: F401  (needed to unpickle CriticalThresholdClassifier)
from config import LOG_DIR, LOW_CONFIDENCE

MODEL_PATH = os.path.join('ml_models', 'best_ticket_model.pkl')

# Prediction log for monitoring (Phase 5) || Bitácora de predicciones para monitoreo (Fase 5)
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('triage')
if not logger.handlers:
    handler = logging.FileHandler(os.path.join(LOG_DIR, 'predictions.log'), encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _load_model():
    """Loads the pipeline; if it fails the app keeps working and tickets go to manual review.
    || Carga el pipeline; si falla, la app sigue funcionando y los tickets quedan para revisión manual."""
    try:
        loaded = joblib.load(MODEL_PATH)
        with open(MODEL_PATH, 'rb') as f:
            version = hashlib.sha256(f.read()).hexdigest()[:12]
        return loaded, version, None
    except Exception as exc:  # noqa: BLE001
        logger.error('MODEL_LOAD_FAILED %s', exc)
        return None, None, str(exc)


model, MODEL_VERSION, MODEL_ERROR = _load_model()


def predict_priority(title, description):
    """Returns (priority, confidence, needs_review) || Devuelve (prioridad, confianza, requiere_revision)"""
    if model is None:
        return None, None, True
    text = f'{title}. {description}'  # Same input as training || Mismo texto que en el entrenamiento
    try:
        priority = model.predict([text])[0]
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_)
            confidence = float(proba[classes.index(priority)])
        needs_review = confidence is not None and confidence < LOW_CONFIDENCE
        return priority, confidence, needs_review
    except Exception as exc:  # noqa: BLE001
        logger.error('PREDICTION_FAILED %s', exc)
        return None, None, True


def classify_and_assign_ticket(title, description, category, client_id):
    # Predict priority using the ML pipeline || Predecir prioridad usando el pipeline de ML
    predicted_priority, confidence, needs_review = predict_priority(title, description)

    # Determine required consultant role by category || Determinar el rol de consultor requerido según categoría
    # Incidents go to Technical Consultants, Requirements go to Functional Consultants
    # || Incidentes van a Consultores Técnicos, Requerimientos a Consultores Funcionales
    target_role = "Consultor Tecnico" if category == "Incidente" else "Consultor Funcional"

    # Select the consultant with the lowest workload (load balancing)
    # || Seleccionar al consultor con menor carga de trabajo (balanceo de carga)
    available_consultants = User.query.filter_by(role=target_role).order_by(User.workload.asc()).all()
    assigned_user = available_consultants[0] if available_consultants else None

    # Create the new ticket record || Crear el nuevo registro de ticket
    new_ticket = Ticket(
        title=title,
        description=description,
        category=category,
        priority=predicted_priority,
        priority_predicted=predicted_priority,
        priority_confidence=confidence,
        model_version=MODEL_VERSION,
        needs_review=needs_review,
        client_id=client_id,
        assigned_to=assigned_user.id if assigned_user else None
    )

    # Increment workload if assigned || Incrementar carga de trabajo si fue asignado
    if assigned_user:
        assigned_user.workload += 1

    db.session.add(new_ticket)
    db.session.commit()

    logger.info('PREDICTION ticket=%s priority=%s confidence=%s needs_review=%s model=%s assigned=%s',
                new_ticket.id, predicted_priority,
                f'{confidence:.3f}' if confidence is not None else 'NA',
                needs_review, MODEL_VERSION, assigned_user.username if assigned_user else 'none')
    return new_ticket
