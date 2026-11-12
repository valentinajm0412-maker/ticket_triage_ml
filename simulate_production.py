"""
Simulador de operación para probar el monitoreo (Fase 6) || Production simulator for monitoring tests

Crea tickets de las últimas 8 semanas con el usuario 'cliente_sim'. Los consultores "revisan"
una parte con la prioridad correcta, toman y resuelven casos. NO usar en producción.

Uso:
    py simulate_production.py            # 160 tickets normales
    py simulate_production.py --drift    # las últimas 3 semanas llegan sistemas nuevos (degradación)
    py simulate_production.py --reset    # borra todo lo simulado
"""
import random
import sys
from datetime import datetime, timedelta

import generate_dataset as gd
from app import app, bcrypt
from models import db, Ticket, User
from triage_service import classify_and_assign_ticket

SIM_USER = 'cliente_sim'
N_TICKETS = 160
WEEKS = 8
REVIEW_RATE = 0.6
# Sistemas que no existen en los datos de entrenamiento (para simular drift)
NEW_MODULES = ['telemedicina', 'laboratorio clinico', 'glosas', 'imagenes diagnosticas', 'portal del medico']


def reset():
    user = User.query.filter_by(username=SIM_USER).first()
    if not user:
        print('No hay datos simulados.')
        return
    tickets = Ticket.query.filter_by(client_id=user.id).all()
    for t in tickets:
        if t.status != 'Resuelto' and t.assignee and t.assignee.workload > 0:
            t.assignee.workload -= 1
        db.session.delete(t)
    db.session.delete(user)
    db.session.commit()
    print(f'{len(tickets)} tickets simulados eliminados.')


def simulate(drift):
    random.seed(7)
    user = User.query.filter_by(username=SIM_USER).first()
    if not user:
        user = User(username=SIM_USER, role='Cliente',
                    password_hash=bcrypt.generate_password_hash('sim-no-login').decode('utf-8'))
        db.session.add(user)
        db.session.commit()
    admin = User.query.filter_by(role='Admin').first()
    now = datetime.now()
    original_modules = list(gd.MODULOS)

    for i in range(N_TICKETS):
        created = now - timedelta(days=random.uniform(0, WEEKS * 7))
        drifted = drift and created > now - timedelta(weeks=3)
        gd.MODULOS = NEW_MODULES if drifted else original_modules
        gen = gd._incidente if random.random() < (0.75 if drifted else 0.6) else gd._requerimiento
        title, desc, category, true_priority = gen()
        desc = gd._ruido(desc)

        t = classify_and_assign_ticket(title, desc, category, user.id)
        t.created_at = created
        if random.random() < REVIEW_RATE:
            t.priority = true_priority
            t.priority_reviewed_by = t.assigned_to or (admin.id if admin else None)
            t.priority_reviewed_at = created + timedelta(hours=1)
            t.needs_review = False
        age_days = (now - created).days
        if age_days > 2 or t.priority == 'Critico':
            t.status = 'En Proceso'
            t.started_at = created + timedelta(minutes=random.uniform(5, 25 if not drifted else 70))
            if age_days > 5:
                t.status = 'Resuelto'
                t.resolved_at = t.started_at + timedelta(hours=random.uniform(1, 30))
                if t.assignee and t.assignee.workload > 0:
                    t.assignee.workload -= 1
        db.session.commit()

    gd.MODULOS = original_modules
    print(f'{N_TICKETS} tickets simulados creados{" (con drift en las últimas 3 semanas)" if drift else ""}.')
    print('Abre CRISP-ML > Monitoreo y Mantenimiento. Para borrarlos: py simulate_production.py --reset')


if __name__ == '__main__':
    with app.app_context():
        if '--reset' in sys.argv:
            reset()
        else:
            simulate('--drift' in sys.argv)
