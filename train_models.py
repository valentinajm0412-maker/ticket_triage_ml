"""
Entrenamiento de modelos || Model training

Fase 2 (data_pipeline.py): selección, limpieza y transformación de datos.
Fase 3 (model_engineering.py): selección de algoritmo con validación cruzada,
ajuste de hiperparámetros y umbral de decisión para tickets críticos.

El test (20%) se separa al inicio y NO participa en ninguna decisión.
"""
import json
import os

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_pipeline import ML_DIR, REPORT_PATH, environment_info, prepare_dataset, save_report
from model_engineering import CRITICAL_LABEL, CV_FOLDS, SEED, _critical_recall, select_and_train

PRIORITY_MODEL_PATH = os.path.join(ML_DIR, 'best_ticket_model.pkl')
CATEGORY_MODEL_PATH = os.path.join(ML_DIR, 'best_category_model.pkl')
MODEL_REPORT_PATH = os.path.join(ML_DIR, 'model_report.json')
TEST_SPLIT_PATH = os.path.join(ML_DIR, 'test_split.csv')  # para la Fase 4

# Fase 2: datos limpios
df, report = prepare_dataset()
X = df['texto']  # titulo + ". " + descripcion (el Pipeline lo limpia)

# Separar test ANTES de cualquier ajuste (evita data leakage)
X_train, X_test, yp_train, yp_test, yc_train, yc_test = train_test_split(
    X, df['prioridad'], df['categoria'], test_size=0.2, random_state=SEED, stratify=df['prioridad'])

# Fase 3: ingeniería del modelo (todo dentro de train)
priority_model, priority_summary = select_and_train('Prioridad', X_train, yp_train, business_rule=True)
category_model, category_summary = select_and_train('Categoría', X_train, yc_train)

joblib.dump(priority_model, PRIORITY_MODEL_PATH)
joblib.dump(category_model, CATEGORY_MODEL_PATH)

old_vectorizer = os.path.join(ML_DIR, 'tfidf_vectorizer.pkl')
if os.path.exists(old_vectorizer):
    os.remove(old_vectorizer)

# Guardar el test intacto para la Fase 4 (Evaluación)
df.loc[X_test.index, ['titulo', 'descripcion', 'texto', 'categoria', 'prioridad']].to_csv(
    TEST_SPLIT_PATH, index=False)

# Referencia rápida en test (la evaluación completa es la Fase 4)
yp_pred = priority_model.predict(X_test)
priority_summary['test_referencia'] = {
    'accuracy': round(accuracy_score(yp_test, yp_pred), 4),
    'f1_macro': round(f1_score(yp_test, yp_pred, average='macro'), 4),
    'recall_critico': round(_critical_recall(yp_test, yp_pred), 4)}
yc_pred = category_model.predict(X_test)
category_summary['test_referencia'] = {
    'accuracy': round(accuracy_score(yc_test, yc_pred), 4),
    'f1_macro': round(f1_score(yc_test, yc_pred, average='macro'), 4)}

# Documentación Fase 2
pipe = priority_model.pipeline if hasattr(priority_model, 'pipeline') else priority_model
feats = pipe.named_steps['caracteristicas']
report['transformacion']['vocabulario'] = {name: len(vec.vocabulary_) for name, vec in feats.transformer_list}
report['particion'] = {'semilla': SEED, 'test_size': 0.2, 'estratificado_por': 'prioridad',
                       'train': len(X_train), 'test': len(X_test)}
report['modelos'] = {
    'prioridad': {'mejor': priority_summary['seleccionado'],
                  'accuracy': priority_summary['test_referencia']['accuracy']},
    'categoria': {'mejor': category_summary['seleccionado'],
                  'accuracy': category_summary['test_referencia']['accuracy']},
    'archivos': [PRIORITY_MODEL_PATH, CATEGORY_MODEL_PATH]}
save_report(report)

# Documentación Fase 3
model_report = {
    'fecha': report['fecha'], 'entorno': environment_info(), 'semilla': SEED,
    'validacion': f'StratifiedKFold {CV_FOLDS} folds sobre train ({len(X_train)} filas)',
    'test_reservado': f'{len(X_test)} filas en {TEST_SPLIT_PATH}',
    'etiqueta_critica': CRITICAL_LABEL,
    'prioridad': priority_summary, 'categoria': category_summary,
    'archivos': [PRIORITY_MODEL_PATH, CATEGORY_MODEL_PATH],
}
with open(MODEL_REPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(model_report, f, ensure_ascii=False, indent=2)

print(f'\nModelos guardados en {ML_DIR}/. Reportes: {REPORT_PATH} y {MODEL_REPORT_PATH}')
