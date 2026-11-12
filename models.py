from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

# Initialize the database || Inicializamos la base de datos
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Roles: 'Admin', 'Technical Consultant', 'Functional Consultant', 'Client' || Roles: 'Admin', 'Consultor Tecnico', 'Consultor Funcional', 'Cliente'
    role = db.Column(db.String(50), nullable=False) 
    
    # Active tickets counter to avoid overloading consultants || Contador de tickets activos para evitar sobrecargar a los consultores
    workload = db.Column(db.Integer, default=0) 

class Ticket(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    # Category: 'Requirement' or 'Incident' || Categoría: 'Requerimiento' o 'Incidente'
    category = db.Column(db.String(50), nullable=False)
    
    # Priority: 'Critical', 'High', 'Medium', 'Low' (To be assigned by ML) || Prioridad: 'Critico', 'Alto', 'Medio', 'Bajo' (Se asignará por ML)
    priority = db.Column(db.String(20), nullable=True) 
    
    # Open, In Progress, Resolved || Abierto, En Proceso, Resuelto
    status = db.Column(db.String(20), default='Abierto') 
    created_at = db.Column(db.DateTime, default=datetime.now)  # local time || hora local
    # Phase 6: response times for business KPIs || Fase 6: tiempos de atención para los KPI de negocio
    started_at = db.Column(db.DateTime, nullable=True)   # Abierto -> En Proceso
    resolved_at = db.Column(db.DateTime, nullable=True)  # -> Resuelto
    
    # Relationships with the User table || Relaciones con la tabla User
    client_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    # Virtual navigation for SQLAlchemy || Navegación virtual para SQLAlchemy
    client = db.relationship('User', foreign_keys=[client_id], backref='tickets_created')
    assignee = db.relationship('User', foreign_keys=[assigned_to], backref='tickets_assigned')

    # ML prediction metadata (Phase 5: deployment) || Metadatos de la predicción (Fase 5: despliegue)
    # 'priority' is the effective priority; 'priority_predicted' keeps what the model said
    # || 'priority' es la prioridad vigente; 'priority_predicted' guarda lo que dijo el modelo
    priority_predicted = db.Column(db.String(20), nullable=True)
    priority_confidence = db.Column(db.Float, nullable=True)
    model_version = db.Column(db.String(40), nullable=True)
    needs_review = db.Column(db.Boolean, default=False)

    # Human feedback: consultant confirms or corrects the priority || Retroalimentación: el consultor confirma o corrige
    priority_reviewed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    priority_reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewer = db.relationship('User', foreign_keys=[priority_reviewed_by])


class ExpertReview(db.Model):
    """Phase 4: domain expert verdict on a test-set error || Fase 4: veredicto de un experto sobre un error del test"""
    id = db.Column(db.Integer, primary_key=True)
    sample_key = db.Column(db.String(32), nullable=False, index=True)  # hash of the text || huella del texto
    text = db.Column(db.Text, nullable=False)
    label_true = db.Column(db.String(20), nullable=False)
    label_predicted = db.Column(db.String(20), nullable=False)
    # 'modelo' = model is wrong, 'etiqueta' = dataset label is wrong, 'ambiguo' = both are defensible
    verdict = db.Column(db.String(20), nullable=False)
    correct_priority = db.Column(db.String(20), nullable=True)
    comment = db.Column(db.Text, nullable=True)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)  # local time || hora local
    reviewer = db.relationship('User')


def ensure_schema():
    """Adds new columns to an existing SQLite DB (create_all does not alter tables).
    || Agrega columnas nuevas a una BD existente (create_all no modifica tablas)."""
    from sqlalchemy import inspect, text
    inspector = inspect(db.engine)
    if 'ticket' not in inspector.get_table_names():
        return
    existing = {c['name'] for c in inspector.get_columns('ticket')}
    new_columns = {
        'priority_predicted': 'VARCHAR(20)', 'priority_confidence': 'FLOAT',
        'model_version': 'VARCHAR(40)', 'needs_review': 'BOOLEAN DEFAULT 0',
        'priority_reviewed_by': 'INTEGER', 'priority_reviewed_at': 'DATETIME',
        'started_at': 'DATETIME', 'resolved_at': 'DATETIME',
    }
    with db.engine.begin() as conn:
        for name, sql_type in new_columns.items():
            if name not in existing:
                conn.execute(text(f'ALTER TABLE ticket ADD COLUMN {name} {sql_type}'))