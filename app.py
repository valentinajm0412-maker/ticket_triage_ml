from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import text

import config
from models import db, User, Ticket, ExpertReview, ensure_schema
from triage_service import classify_and_assign_ticket, MODEL_VERSION, MODEL_ERROR, model as triage_model
from crisp_fase1 import build_context as crisp_fase1_context
from crisp_fase2 import build_context as crisp_fase2_context
from crisp_fase3 import build_context as crisp_fase3_context
from crisp_fase4 import build_context as crisp_fase4_context
from evaluation import load_evaluation
from monitoring import build_context as crisp_fase6_context
from deployment import (deployment_checklist, feedback_stats, export_feedback,
                        TRAINING_GUIDE, INCIDENT_PROTOCOL)

app = Flask(__name__)

# Base configuration from environment variables (Phase 5) || Configuración desde variables de entorno (Fase 5)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize extensions || Inicializar extensiones
db.init_app(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)  # Protects every POST form || Protege todos los formularios POST
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()
    ensure_schema()  # Adds new columns to existing databases || Agrega columnas nuevas a BD existentes

# Load user for Flask-Login || Cargar usuario para Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def admin_required(view):
    """Only Admin can open CRISP-ML pages || Solo el Admin puede abrir las hojas CRISP-ML"""
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if current_user.role != 'Admin':
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)
    return wrapper


def can_manage(ticket):
    return current_user.role == 'Admin' or ticket.assigned_to == current_user.id


# Login Route || Ruta de Inicio de Sesión
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Credenciales incorrectas || Invalid credentials')

    return render_template('login.html')

# Dashboard Route with ticket list || Ruta del Tablero con listado de tickets
@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    # Role-based ticket display || Visualización de tickets basada en rol
    if current_user.role == 'Admin':
        tickets = Ticket.query.order_by(Ticket.created_at.desc()).all()
    elif current_user.role == 'Cliente':
        tickets = Ticket.query.filter_by(client_id=current_user.id).order_by(Ticket.created_at.desc()).all()
    else:
        tickets = Ticket.query.filter_by(assigned_to=current_user.id).order_by(Ticket.created_at.desc()).all()

    return render_template('dashboard.html', tickets=tickets, priorities=config.VALID_PRIORITIES)

# Route to create tickets || Ruta para crear tickets
@app.route('/create_ticket', methods=['GET', 'POST'])
@login_required
def create_ticket():
    if current_user.role not in ['Cliente', 'Admin']:
        flash('Solo clientes o administradores pueden crear casos || Only clients or admins can create tickets')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        title = (request.form.get('title') or '').strip()
        description = (request.form.get('description') or '').strip()
        category = request.form.get('category')
        if not title or not description or category not in ['Incidente', 'Requerimiento']:
            flash('Completa título, descripción y tipo de caso || Fill in all fields')
            return redirect(url_for('create_ticket'))

        classify_and_assign_ticket(title, description, category, current_user.id)
        return redirect(url_for('dashboard'))

    return render_template('create_ticket.html')

# Route to update ticket status (POST + validated status) || Ruta para actualizar estado (POST + estado validado)
@app.route('/update_ticket/<int:ticket_id>', methods=['POST'])
@login_required
def update_ticket(ticket_id):
    ticket = db.session.get(Ticket, ticket_id)
    new_status = request.form.get('status')
    if not ticket or new_status not in config.VALID_STATUSES or not can_manage(ticket):
        return redirect(url_for('dashboard'))

    # Balance workload if ticket is resolved || Balancear carga de trabajo si se resuelve el ticket
    if new_status == 'Resuelto' and ticket.status != 'Resuelto':
        if ticket.assignee and ticket.assignee.workload > 0:
            ticket.assignee.workload -= 1
    elif new_status != 'Resuelto' and ticket.status == 'Resuelto':
        if ticket.assignee:
            ticket.assignee.workload += 1

    # Response times for Phase 6 KPIs || Tiempos de atención para los KPI de la Fase 6
    if new_status == 'En Proceso' and not ticket.started_at:
        ticket.started_at = datetime.now()
    if new_status == 'Resuelto':
        ticket.resolved_at = datetime.now()
        ticket.started_at = ticket.started_at or ticket.resolved_at

    ticket.status = new_status
    db.session.commit()
    return redirect(url_for('dashboard'))

# Consultant confirms or corrects the ML priority (feedback loop) || El consultor confirma o corrige la prioridad
@app.route('/ticket/<int:ticket_id>/priority', methods=['POST'])
@login_required
def review_priority(ticket_id):
    ticket = db.session.get(Ticket, ticket_id)
    new_priority = request.form.get('priority')
    if not ticket or new_priority not in config.VALID_PRIORITIES or not can_manage(ticket):
        return redirect(url_for('dashboard'))

    ticket.priority = new_priority
    ticket.priority_reviewed_by = current_user.id
    ticket.priority_reviewed_at = datetime.now()
    ticket.needs_review = False
    db.session.commit()
    action = 'confirmada' if new_priority == ticket.priority_predicted else 'corregida'
    flash(f'Prioridad del ticket #{ticket.id} {action}: {new_priority}')
    return redirect(url_for('dashboard'))

# Admin Route: Manage users and reset passwords || Ruta Admin: Gestionar usuarios y restablecer contraseñas
@app.route('/admin/users', methods=['GET', 'POST'])
@admin_required
def admin_users():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        new_password = request.form.get('new_password')
        target_user = db.session.get(User, int(user_id))

        if target_user and new_password:
            # Hash new password || Encriptar nueva contraseña
            target_user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
            db.session.commit()
            flash(f'Contraseña actualizada para {target_user.username} || Password updated for {target_user.username}')

    users = User.query.all()
    return render_template('admin_users.html', users=users)

# CRISP-ML Phase 1 page || Hoja Fase 1 CRISP-ML
@app.route('/crisp/fase1')
@admin_required
def crisp_fase1():
    return render_template('crisp_fase1.html', **crisp_fase1_context(Ticket))

# CRISP-ML Phase 2 page || Hoja Fase 2 CRISP-ML
@app.route('/crisp/fase2')
@admin_required
def crisp_fase2():
    return render_template('crisp_fase2.html', **crisp_fase2_context())

# CRISP-ML Phase 3 page || Hoja Fase 3 CRISP-ML
@app.route('/crisp/fase3')
@admin_required
def crisp_fase3():
    return render_template('crisp_fase3.html', **crisp_fase3_context())

# CRISP-ML Phase 4 page || Hoja Fase 4 CRISP-ML
@app.route('/crisp/fase4')
@admin_required
def crisp_fase4():
    return render_template('crisp_fase4.html', **crisp_fase4_context(ExpertReview))

# Phase 4: save an expert verdict || Fase 4: guardar el veredicto de un experto
@app.route('/crisp/fase4/review', methods=['POST'])
@admin_required
def crisp_fase4_review():
    ev = load_evaluation()
    key = request.form.get('key')
    verdict = request.form.get('verdict')
    error = next((e for e in (ev or {}).get('errores', []) if e['key'] == key), None)
    if error and verdict in ['modelo', 'etiqueta', 'ambiguo']:
        correct = request.form.get('correct_priority')
        db.session.add(ExpertReview(
            sample_key=key, text=error['texto'], label_true=error['real'],
            label_predicted=error['predicho'], verdict=verdict,
            correct_priority=correct if correct in config.VALID_PRIORITIES else None,
            comment=(request.form.get('comment') or '').strip() or None,
            reviewer_id=current_user.id))
        db.session.commit()
        flash('Revisión guardada.')
    return redirect(url_for('crisp_fase4'))

# CRISP-ML Phase 5 page || Hoja Fase 5 CRISP-ML
@app.route('/crisp/fase5')
@admin_required
def crisp_fase5():
    admin = User.query.filter_by(username='admin').first()
    return render_template(
        'crisp_fase5.html',
        checklist=deployment_checklist(app, admin, bcrypt, MODEL_ERROR, MODEL_VERSION),
        stats=feedback_stats(Ticket), guide=TRAINING_GUIDE, protocol=INCIDENT_PROTOCOL,
        model_version=MODEL_VERSION, low_confidence=config.LOW_CONFIDENCE,
        app_env=config.APP_ENV, evaluation=load_evaluation())

# Phase 5: export consultant feedback as real training data || Fase 5: exportar retroalimentación como datos reales
@app.route('/crisp/fase5/export', methods=['POST'])
@admin_required
def crisp_fase5_export():
    n, path = export_feedback(Ticket)
    flash(f'{n} tickets exportados a {path}. Ejecuta py train_models.py para reentrenar.'
          if n else 'No hay tickets revisados por consultores todavía.')
    return redirect(url_for('crisp_fase5'))

# CRISP-ML Phase 6 page || Hoja Fase 6 CRISP-ML
@app.route('/crisp/fase6')
@admin_required
def crisp_fase6():
    return render_template('crisp_fase6.html', **crisp_fase6_context(Ticket, User, triage_model, MODEL_VERSION))

# Health check for monitoring (no login) || Verificación de salud para monitoreo (sin login)
@app.route('/health')
def health():
    try:
        db.session.execute(text('SELECT 1'))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    status = 'ok' if db_ok and triage_model is not None else 'degraded'
    return jsonify({'status': status, 'database': db_ok, 'model_loaded': triage_model is not None,
                    'model_version': MODEL_VERSION, 'environment': config.APP_ENV}), \
        200 if status == 'ok' else 503

# Logout Route || Ruta de Cerrar Sesión
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Development server only; production uses run_production.py || Solo desarrollo; producción usa run_production.py
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
