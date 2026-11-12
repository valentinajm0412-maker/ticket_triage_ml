import joblib
from models import db, User, Ticket

import data_pipeline  # noqa: F401  (needed to unpickle TextCleaner || necesario para cargar TextCleaner)
import model_engineering  # noqa: F401  (needed to unpickle CriticalThresholdClassifier)

# Load the full pipeline: cleaning + features + model || Cargar el pipeline completo: limpieza + características + modelo
model = joblib.load('ml_models/best_ticket_model.pkl')

def classify_and_assign_ticket(title, description, category, client_id):
    # Predict priority using the ML pipeline || Predecir prioridad usando el pipeline de ML
    # Same input as training: title + description || Mismo texto que en el entrenamiento
    predicted_priority = model.predict([f'{title}. {description}'])[0]

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
        client_id=client_id,
        assigned_to=assigned_user.id if assigned_user else None
    )

    # Increment workload if assigned || Incrementar carga de trabajo si fue asignado
    if assigned_user:
        assigned_user.workload += 1

    db.session.add(new_ticket)
    db.session.commit()

    return new_ticket