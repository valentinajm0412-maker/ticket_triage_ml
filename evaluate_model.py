"""
CRISP-ML Fase 4: ejecuta la evaluación sobre el test reservado.
|| CRISP-ML Phase 4: runs the evaluation on the held-out test set.

Uso: py evaluate_model.py   (también se ejecuta automáticamente al final de train_models.py)
"""
from evaluation import EVALUATION_REPORT_PATH, run_evaluation

if __name__ == '__main__':
    r = run_evaluation()
    m = r['metricas_test']
    print('=== EVALUACIÓN EN TEST ===')
    print(f"Modelo: {r['modelo']} ({r['test_filas']} tickets de test)")
    print(f"Exactitud {m['accuracy']:.1%} | F1 macro {m['f1_macro']:.1%} | Recall Crítico {m['recall_critico']:.1%}")
    for c in r['criterios']:
        print(f"  [{'OK' if c['cumple'] else 'NO'}] {c['nombre']}: {c['valor']:.1%} (mínimo {c['minimo']:.0%})")
    for k in r['riesgos']:
        print(f"  Riesgo {k['riesgo']}: {k['nivel']}")
    print(f"Decisión: {r['decision']}")
    print(f"Errores para revisión de expertos: {len(r['errores'])}")
    print(f'Reporte guardado en {EVALUATION_REPORT_PATH}')
