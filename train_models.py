"""
Entrenamiento de modelos || Model training

Usa data_pipeline.py (Fase 2: Ingeniería de Datos) y guarda cada modelo como un
Pipeline completo: limpieza de texto -> características TF-IDF -> clasificador.
Así la app transforma el texto exactamente igual que en el entrenamiento.
"""
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_pipeline import ML_DIR, REPORT_PATH, TextCleaner, build_features, prepare_dataset, save_report

SEED = 42
PRIORITY_MODEL_PATH = os.path.join(ML_DIR, 'best_ticket_model.pkl')
CATEGORY_MODEL_PATH = os.path.join(ML_DIR, 'best_category_model.pkl')

# 1-2. Selección y limpieza (Fase 2)
df, report = prepare_dataset()
X = df['texto']  # titulo + ". " + descripcion (el Pipeline lo limpia)

# 3. Dividir ANTES de ajustar cualquier transformación (evita data leakage)
X_train, X_test, yp_train, yp_test, yc_train, yc_test = train_test_split(
    X, df['prioridad'], df['categoria'], test_size=0.2, random_state=SEED, stratify=df['prioridad'])


def train_best_classifier(target_name, y_train, y_test):
    candidates = {
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(random_state=SEED, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=2000),
    }
    results, best = {}, (None, -1, '')
    print(f'=== Entrenando modelos para: {target_name.upper()} ===\n')

    for name, clf in candidates.items():
        pipe = Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                         ('modelo', clf)])
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        results[name] = round(acc, 4)
        print(f'{name} Accuracy: {acc:.4f}')
        if acc > best[1]:
            best = (pipe, acc, name)

    pipe, acc, name = best
    print(f'\nMejor modelo para {target_name}: {name} ({acc:.4f})')
    print(classification_report(y_test, pipe.predict(X_test)))
    return pipe, {'mejor': name, 'accuracy': round(acc, 4), 'candidatos': results}


# 4. Entrenar y guardar Pipelines completos
priority_pipe, priority_info = train_best_classifier('Prioridad', yp_train, yp_test)
category_pipe, category_info = train_best_classifier('Categoría', yc_train, yc_test)

joblib.dump(priority_pipe, PRIORITY_MODEL_PATH)
joblib.dump(category_pipe, CATEGORY_MODEL_PATH)

# El vectorizador ya va dentro del Pipeline; se elimina el archivo antiguo si existe
old_vectorizer = os.path.join(ML_DIR, 'tfidf_vectorizer.pkl')
if os.path.exists(old_vectorizer):
    os.remove(old_vectorizer)

# 5. Documentación: completar y guardar el reporte
feats = priority_pipe.named_steps['caracteristicas']
report['transformacion']['vocabulario'] = {
    name: len(vec.vocabulary_) for name, vec in feats.transformer_list}
report['particion'] = {'semilla': SEED, 'test_size': 0.2, 'estratificado_por': 'prioridad',
                       'train': len(X_train), 'test': len(X_test)}
report['modelos'] = {'prioridad': priority_info, 'categoria': category_info,
                     'archivos': [PRIORITY_MODEL_PATH, CATEGORY_MODEL_PATH]}
save_report(report)

print(f'Pipelines guardados en {ML_DIR}/ y reporte en {REPORT_PATH}')
