"""
Exporta los tickets revisados por consultores a ml_models/real_tickets.csv
para usarlos como datos reales en el próximo entrenamiento (Fase 2 -> fuente 'real').

Uso: py export_feedback.py   y luego   py train_models.py
"""
from app import app
from deployment import export_feedback
from models import Ticket

if __name__ == '__main__':
    with app.app_context():
        n, path = export_feedback(Ticket)
    if n:
        print(f'{n} tickets revisados exportados a {path}. Ahora ejecuta: py train_models.py')
    else:
        print('No hay tickets revisados por consultores todavía.')
