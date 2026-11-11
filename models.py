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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships with the User table || Relaciones con la tabla User
    client_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    # Virtual navigation for SQLAlchemy || Navegación virtual para SQLAlchemy
    client = db.relationship('User', foreign_keys=[client_id], backref='tickets_created')
    assignee = db.relationship('User', foreign_keys=[assigned_to], backref='tickets_assigned')