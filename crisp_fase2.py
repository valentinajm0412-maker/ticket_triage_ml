"""
Hoja CRISP-ML Fase 2: Ingeniería de Datos || CRISP-ML Phase 2 page: Data Engineering
Lee ml_models/data_report.json (lo genera train_models.py) y verifica el entorno actual.
"""
from data_pipeline import REPORT_PATH, environment_info, load_report


def build_context():
    report = load_report()
    if report is None:
        return {'report': None, 'report_path': REPORT_PATH}

    current = environment_info()
    trained = report.get('entorno', {})
    env_rows = [{'libreria': k, 'entrenamiento': trained.get(k, '-'), 'actual': v,
                 'ok': trained.get(k) == v or k == 'python'} for k, v in current.items()]

    return {
        'report': report,
        'report_path': REPORT_PATH,
        'env_rows': env_rows,
        'env_ok': all(r['ok'] for r in env_rows),
    }
