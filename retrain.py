"""
Reentrenamiento seguro (Fase 6: Mantenimiento) || Safe retraining

1. Exporta las revisiones de los consultores (ml_models/real_tickets.csv)
2. Respalda el modelo actual en ml_models/backup/
3. Reentrena (train_models.py: Fases 2, 3 y 4)
4. Compara con el modelo anterior: si empeora la detección de críticos o el F1,
   restaura automáticamente el modelo anterior (rollback)

Uso:  py retrain.py            (normal)
      py retrain.py --force    (conserva el modelo nuevo aunque empeore)
      py retrain.py --no-export
Después, reinicia la app para que cargue el modelo nuevo.
"""
import json
import os
import shutil
import subprocess
import sys

from data_pipeline import ML_DIR
from evaluation import load_evaluation
from monitoring import HISTORY_PATH, load_history

BACKUP_DIR = os.path.join(ML_DIR, 'backup')
FILES = ['best_ticket_model.pkl', 'best_category_model.pkl', 'data_report.json', 'model_report.json',
         'evaluation_report.json', 'test_split.csv', 'ticket_dataset_clean.csv', 'model_history.json']
TOLERANCE = 0.02  # caída máxima aceptada en recall de Crítico y F1 macro


def backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name in FILES:
        src = os.path.join(ML_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BACKUP_DIR, name))


def restore():
    for name in FILES:
        src = os.path.join(BACKUP_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ML_DIR, name))


def main():
    force = '--force' in sys.argv
    if '--no-export' not in sys.argv:
        from app import app
        from deployment import export_feedback
        from models import Ticket
        with app.app_context():
            n, path = export_feedback(Ticket)
        print(f'1. Retroalimentación exportada: {n} tickets revisados -> {path}')

    before = load_evaluation()
    backup()
    print(f'2. Modelo actual respaldado en {BACKUP_DIR}')

    print('3. Reentrenando (1-3 minutos)...')
    result = subprocess.run([sys.executable, 'train_models.py'])
    if result.returncode != 0:
        restore()
        print('ERROR: el entrenamiento falló. Se restauró el modelo anterior.')
        sys.exit(1)

    after = load_evaluation()
    if not before:
        print('4. No había modelo anterior para comparar. Se conserva el nuevo.')
        return

    b, a = before['metricas_test'], after['metricas_test']
    print('4. Comparación (test):')
    for k, label in [('recall_critico', 'Recall Crítico'), ('f1_macro', 'F1 macro'), ('accuracy', 'Exactitud')]:
        print(f'   {label:15} {b[k]:.1%} -> {a[k]:.1%} ({a[k] - b[k]:+.1%})')

    worse = [k for k in ('recall_critico', 'f1_macro') if a[k] < b[k] - TOLERANCE]
    if worse and not force:
        new_entry = (load_history() or [{}])[-1]
        restore()
        history = load_history()
        new_entry['decision'] = f'Rechazado (rollback): empeoró {", ".join(worse)}'
        history.append(new_entry)
        with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f'RESULTADO: el modelo nuevo empeoró ({", ".join(worse)}). Se restauró el anterior.')
        print('Usa --force si aun así quieres conservar el nuevo.')
    else:
        print('RESULTADO: modelo nuevo aceptado. Reinicia la app (Ctrl + C y py app.py) para usarlo.')


if __name__ == '__main__':
    main()
