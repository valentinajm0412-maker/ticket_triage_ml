from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from models import db, User, Ticket
from triage_service import classify_and_assign_ticket
from crisp_fase1 import build_context as crisp_fase1_context

app = Flask(__name__)

# Base configuration || Configuración base
app.config['SECRET_KEY'] = 'clave_secreta_ticket_ml_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ticket_triage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions || Inicializar extensiones
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

# Load user for Flask-Login || Cargar usuario para Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

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

    return render_template('dashboard.html', tickets=tickets)

# Route to create tickets || Ruta para crear tickets
@app.route('/create_ticket', methods=['GET', 'POST'])
@login_required
def create_ticket():
    if current_user.role not in ['Cliente', 'Admin']:
        flash('Solo clientes o administradores pueden crear casos || Only clients or admins can create tickets')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category')

        classify_and_assign_ticket(title, description, category, current_user.id)
        return redirect(url_for('dashboard'))

    return render_template('create_ticket.html')

# Route to update ticket status || Ruta para actualizar el estado del ticket
@app.route('/update_ticket/<int:ticket_id>/<string:new_status>')
@login_required
def update_ticket(ticket_id, new_status):
    ticket = db.session.get(Ticket, ticket_id)
    if not ticket:
        return redirect(url_for('dashboard'))

    # Verify authorization (only assignee or admin) || Verificar autorización (solo asignado o admin)
    if current_user.role != 'Admin' and ticket.assigned_to != current_user.id:
        return redirect(url_for('dashboard'))

    # Balance workload if ticket is resolved || Balancear carga de trabajo si se resuelve el ticket
    if new_status == 'Resuelto' and ticket.status != 'Resuelto':
        if ticket.assignee and ticket.assignee.workload > 0:
            ticket.assignee.workload -= 1
    elif new_status != 'Resuelto' and ticket.status == 'Resuelto':
        if ticket.assignee:
            ticket.assignee.workload += 1

    ticket.status = new_status
    db.session.commit()
    return redirect(url_for('dashboard'))

# Admin Route: Manage users and reset passwords || Ruta Admin: Gestionar usuarios y restablecer contraseñas
@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
def admin_users():
    if current_user.role != 'Admin':
        return redirect(url_for('dashboard'))

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

# CRISP-ML Phase 1 page (Admin only) || Hoja Fase 1 CRISP-ML (solo Admin)
@app.route('/crisp/fase1')
@login_required
def crisp_fase1():
    if current_user.role != 'Admin':
        return redirect(url_for('dashboard'))
    return render_template('crisp_fase1.html', **crisp_fase1_context(Ticket))

# Logout Route || Ruta de Cerrar Sesión
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)