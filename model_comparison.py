"""
Comparación de los modelos del curso (Fase 4: Evaluación) || Course models comparison

Cada modelo responde la pregunta de negocio para la que sirve, con el MISMO split train/test
y las MISMAS características de texto (Fase 2) que el modelo en producción:

| Pregunta                                         | Modelo                      | Tipo                   |
| ¿Qué prioridad tiene el ticket?                  | Logistic Regression, KNN    | Clasificación          |
| ¿Qué tan grave es? (puntaje 0-3)                 | Linear Regression           | Regresión              |
| ¿Qué tipos de tickets llegan?                    | K-Means                     | Agrupamiento           |
| ¿Qué hacer con el ticket?                        | Q-Learning                  | Aprendizaje por refuerzo |

Random Forest (modelo en producción, Fase 3) se incluye como referencia.
Uso: py compare_models.py   (también corre al final de train_models.py)
"""
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score, mean_absolute_error,
                             r2_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

import data_pipeline  # noqa: F401  (para cargar los .pkl)
import model_engineering  # noqa: F401
from data_pipeline import CLEAN_PATH, ML_DIR, TextCleaner, build_features
from model_engineering import (CANDIDATES, OrdinalLinearClassifier, apply_threshold, make_pipeline,
                               scores_to_labels)
from sklearn.base import clone

SEED = 42
TEST_SPLIT_PATH = os.path.join(ML_DIR, 'test_split.csv')
PRIORITY_MODEL_PATH = os.path.join(ML_DIR, 'best_ticket_model.pkl')
MODEL_REPORT_PATH = os.path.join(ML_DIR, 'model_report.json')
COMPARISON_PATH = os.path.join(ML_DIR, 'model_comparison.json')

SCORE = {'Bajo': 0, 'Medio': 1, 'Alto': 2, 'Critico': 3}
LABEL = {v: k for k, v in SCORE.items()}


def _clf_metrics(y_true, y_pred):
    return {'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'f1_macro': round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'recall_critico': round(recall_score(y_true, y_pred, labels=['Critico'], average='macro',
                                                 zero_division=0), 4)}


def _load_split():
    test = pd.read_csv(TEST_SPLIT_PATH)
    clean = pd.read_csv(CLEAN_PATH)
    clean['texto'] = clean['titulo'].fillna('') + '. ' + clean['descripcion']
    train = clean[~clean['texto'].isin(set(test['texto']))].reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------- Clasificación
def logistic_regression(train, test, cv):
    pipe = Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                     ('modelo', LogisticRegression(max_iter=3000, random_state=SEED))])
    search = GridSearchCV(pipe, {'modelo__C': [1, 10, 30], 'modelo__class_weight': [None, 'balanced']},
                          scoring='f1_macro', cv=cv, n_jobs=-1).fit(train['texto'], train['prioridad'])
    pred = search.predict(test['texto'])
    # Mismo ajuste de umbral de críticos que el modelo de producción (justo para comparar)
    proba = search.predict_proba(test['texto'])
    with open(MODEL_REPORT_PATH, encoding='utf-8') as f:
        threshold = json.load(f)['prioridad'].get('umbral', {}).get('umbral', 0.5)
    pred_t = apply_threshold(proba, search.classes_, threshold)
    return {'cv_f1_macro': round(search.best_score_, 4), 'parametros': _params(search),
            'test': _clf_metrics(test['prioridad'], pred),
            'test_con_umbral': {**_clf_metrics(test['prioridad'], pred_t), 'umbral': threshold}}


def knn(train, test, cv):
    pipe = Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                     ('modelo', KNeighborsClassifier(metric='cosine', algorithm='brute'))])
    search = GridSearchCV(pipe, {'modelo__n_neighbors': [3, 5, 9, 15, 25],
                                 'modelo__weights': ['uniform', 'distance']},
                          scoring='f1_macro', cv=cv, n_jobs=-1).fit(train['texto'], train['prioridad'])
    return {'cv_f1_macro': round(search.best_score_, 4), 'parametros': _params(search),
            'test': _clf_metrics(test['prioridad'], search.predict(test['texto']))}


def _params(search):
    return {k.replace('modelo__', ''): str(v) for k, v in search.best_params_.items()}


def _dense_features(train, test, n_components=100):
    """TF-IDF -> SVD (LSA) -> normalizado: representación densa para K-Means y Regresión Lineal."""
    feats = Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                      ('svd', TruncatedSVD(n_components=n_components, random_state=SEED)),
                      ('norma', Normalizer())])
    return feats.fit_transform(train['texto']), feats.transform(test['texto']), feats


# ---------------------------------------------------------------- Regresión
def linear_regression(train, test, mr):
    """Prioridad como número ordenado (Bajo=0 ... Crítico=3). Usa el corte de Crítico y la validación
    cruzada de la Fase 3 (model_report.json -> prioridad.ordinal)."""
    ordn = mr['prioridad'].get('ordinal', {'corte': 2.5, 'sigma': 0.5, 'despues': {}, 'antes': {}})
    production = joblib.load(PRIORITY_MODEL_PATH)
    if isinstance(production, OrdinalLinearClassifier):
        model = production  # exactamente el modelo en producción (mismos números que la Evaluación)
    else:
        model = OrdinalLinearClassifier(ordn['corte'], ordn['sigma']).fit(train['texto'], train['prioridad'])
    raw = model.score_ticket(test['texto'])
    yte = test['prioridad'].map(SCORE).to_numpy()
    return {'r2': round(r2_score(yte, raw), 4), 'mae': round(mean_absolute_error(yte, raw), 3),
            'corte_critico': ordn['corte'], 'cv_redondeado': ordn.get('antes', {}),
            'cv_con_corte': ordn.get('despues', {}), 'cv_f1_macro': ordn.get('despues', {}).get('f1_macro'),
            'test_redondeado': _clf_metrics(test['prioridad'], scores_to_labels(raw)),
            'test': _clf_metrics(test['prioridad'], scores_to_labels(raw, ordn['corte']))}


def _parse(v):
    try:
        return None if v == 'None' else (int(v) if v.isdigit() else float(v))
    except ValueError:
        return v


def thresholded_prob_model(train, test, mr):
    """Mejor modelo con probabilidad de la Fase 3 (Random Forest) + su umbral de Crítico, medido en test."""
    u = mr['prioridad']['umbral']
    row = next(r for r in mr['prioridad']['candidatos'] if r['algoritmo'] == u['algoritmo'])
    base, _ = CANDIDATES[u['algoritmo']]
    params = {k: _parse(v) for k, v in row['mejores_parametros'].items()}
    pipe = make_pipeline(clone(base).set_params(**params)).fit(train['texto'], train['prioridad'])
    pred = apply_threshold(pipe.predict_proba(test['texto']), pipe.classes_, u['umbral'])
    return {'algoritmo': u['algoritmo'], 'umbral': u['umbral'], 'cv_f1_macro': u['despues']['f1_macro'],
            'cv_recall_critico': u['despues']['recall_critico'], 'test': _clf_metrics(test['prioridad'], pred)}


# ---------------------------------------------------------------- Agrupamiento
def kmeans(Xtr, Xte, train, test, feats):
    scores = {}
    for k in range(4, 13, 2):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xtr)
        scores[k] = round(silhouette_score(Xtr, km.labels_, sample_size=1500, random_state=SEED), 4)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xtr)
    labels_te = km.predict(Xte)

    # Interpretación: palabras más representativas de cada grupo y su prioridad dominante
    words_vec = dict(feats.named_steps['caracteristicas'].transformer_list)['palabras']
    clean_tr = TextCleaner().transform(train['texto'])
    tfidf_tr = words_vec.transform(clean_tr)
    terms = np.array(words_vec.get_feature_names_out())
    groups = []
    for c in range(best_k):
        mask = km.labels_ == c
        top = terms[np.asarray(tfidf_tr[mask].mean(axis=0)).ravel().argsort()[::-1]]
        top = [t for t in top if ' ' not in t][:6]
        pr = train.loc[mask, 'prioridad'].value_counts(normalize=True)
        cat = train.loc[mask, 'categoria'].value_counts(normalize=True)
        groups.append({'grupo': c + 1, 'tickets': int(mask.sum()), 'palabras': top,
                       'prioridad_dominante': pr.index[0], 'pct_prioridad': round(float(pr.iloc[0]), 3),
                       'categoria_dominante': cat.index[0], 'pct_categoria': round(float(cat.iloc[0]), 3)})

    # ¿Sirve para predecir prioridad? cada grupo -> su prioridad mayoritaria
    mapping = {g['grupo'] - 1: g['prioridad_dominante'] for g in groups}
    pred = [mapping[c] for c in labels_te]
    return {'k': best_k, 'silueta_por_k': scores, 'silueta': scores[best_k],
            'ari_prioridad': round(adjusted_rand_score(test['prioridad'], labels_te), 4),
            'ari_categoria': round(adjusted_rand_score(test['categoria'], labels_te), 4),
            'como_clasificador': _clf_metrics(test['prioridad'], pred), 'grupos': groups}


# ---------------------------------------------------------------- Aprendizaje por refuerzo
ACTIONS = ['Atender de inmediato', 'Cola normal', 'Revisión humana']
# Recompensas (supuestos de negocio, documentados): filas = prioridad REAL
REWARDS = {
    'Critico': {'Atender de inmediato': 10, 'Cola normal': -20, 'Revisión humana': 6},
    'Alto':    {'Atender de inmediato': 3,  'Cola normal': -3,  'Revisión humana': 3},
    'Medio':   {'Atender de inmediato': -3, 'Cola normal': 2,   'Revisión humana': -1},
    'Bajo':    {'Atender de inmediato': -3, 'Cola normal': 2,   'Revisión humana': -1},
}


def _state(pred, conf):
    return f"{pred} · {'alta' if conf >= 0.5 else 'baja'} confianza"


def q_learning(test, episodes=400, alpha=0.05, gamma=0.0):
    """Estado = (prioridad predicha, confianza); acción = qué hacer; recompensa según la prioridad real.
    Se entrena con la mitad del test (predicciones honestas del modelo) y se evalúa con la otra mitad.
    gamma = 0: la acción sobre un ticket no cambia cuál llega después, así que solo cuenta la recompensa inmediata."""
    model = joblib.load(PRIORITY_MODEL_PATH)
    proba = model.predict_proba(test['texto'])
    pred = model.predict(test['texto'])
    conf = proba.max(axis=1)
    states = [_state(p, c) for p, c in zip(pred, conf)]
    true = test['prioridad'].tolist()
    idx = list(range(len(test)))
    rng = random.Random(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    learn_idx, eval_idx = idx[:half], idx[half:]

    all_states = sorted(set(states))
    Q = {s: {a: 0.0 for a in ACTIONS} for s in all_states}
    eps = 1.0
    for _ in range(episodes):
        order = learn_idx[:]
        rng.shuffle(order)
        for j, i in enumerate(order):
            s = states[i]
            a = rng.choice(ACTIONS) if rng.random() < eps else max(Q[s], key=Q[s].get)
            r = REWARDS[true[i]][a]
            s_next = states[order[j + 1]] if j + 1 < len(order) else None
            future = max(Q[s_next].values()) if s_next else 0
            Q[s][a] += alpha * (r + gamma * future - Q[s][a])
        eps = max(0.05, eps * 0.98)

    policy = {s: max(Q[s], key=Q[s].get) for s in all_states}

    def evaluate(rule):
        rewards = [REWARDS[true[i]][rule(i)] for i in eval_idx]
        crit = [i for i in eval_idx if true[i] == 'Critico']
        queued = sum(rule(i) == 'Cola normal' for i in crit)
        return {'recompensa_media': round(float(np.mean(rewards)), 3),
                'criticos_en_cola': int(queued), 'criticos': len(crit)}

    baselines = {
        'Q-Learning (política aprendida)': evaluate(lambda i: policy.get(states[i], 'Revisión humana')),
        'Regla fija: Crítico inmediato, resto en cola': evaluate(
            lambda i: 'Atender de inmediato' if pred[i] == 'Critico' else 'Cola normal'),
        'Todo a revisión humana': evaluate(lambda i: 'Revisión humana'),
        'Todo en cola normal': evaluate(lambda i: 'Cola normal'),
    }
    counts = pd.Series(states).value_counts().to_dict()
    table = [{'estado': s, 'tickets': int(counts.get(s, 0)), 'accion': policy[s],
              'q': {a: round(v, 2) for a, v in Q[s].items()}} for s in all_states]
    return {'politica': table, 'comparacion': baselines, 'recompensas': REWARDS,
            'episodios': episodes, 'alpha': alpha, 'gamma': gamma,
            'tickets_aprendizaje': len(learn_idx), 'tickets_evaluacion': len(eval_idx)}


# ---------------------------------------------------------------- Orquestación
def run_comparison():
    train, test = _load_split()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print('Comparando modelos del curso...')
    lr = logistic_regression(train, test, cv)
    print(f"  Logistic Regression  test acc {lr['test']['accuracy']:.1%}")
    kn = knn(train, test, cv)
    print(f"  KNN                  test acc {kn['test']['accuracy']:.1%}")
    with open(MODEL_REPORT_PATH, encoding='utf-8') as f:
        mr = json.load(f)
    lin = linear_regression(train, test, mr)
    print(f"  Linear Regression    R2 {lin['r2']:.2f}  test acc (con corte) {lin['test']['accuracy']:.1%}")
    rf = thresholded_prob_model(train, test, mr)
    print(f"  {rf['algoritmo']:20} test acc (con umbral) {rf['test']['accuracy']:.1%}")
    Xtr, Xte, feats = _dense_features(train, test)
    km = kmeans(Xtr, Xte, train, test, feats)
    print(f"  K-Means              k={km['k']} silueta {km['silueta']:.2f}")
    ql = q_learning(test)
    print(f"  Q-Learning           recompensa {ql['comparacion']['Q-Learning (política aprendida)']['recompensa_media']}")

    production = mr['prioridad']['seleccionado']

    ranking = sorted([
        {'modelo': f"{rf['algoritmo']} + umbral Crítico {rf['umbral']}", 'clave': 'random_forest',
         'cv_f1_macro': rf['cv_f1_macro'], 'cv_recall_critico': rf['cv_recall_critico'],
         'en_produccion': production.startswith(rf['algoritmo']), **rf['test']},
        {'modelo': f"Linear Regression (puntaje 0-3) + corte Crítico {lin['corte_critico']}", 'clave': 'linear_regression',
         'cv_f1_macro': lin['cv_f1_macro'], 'cv_recall_critico': lin['cv_con_corte'].get('recall_critico'),
         'en_produccion': production.startswith('Linear Regression'), **lin['test']},
        {'modelo': 'Logistic Regression', 'clave': 'logistic_regression', 'cv_f1_macro': lr['cv_f1_macro'],
         'cv_recall_critico': None, 'en_produccion': False, **lr['test']},
        {'modelo': 'KNN', 'clave': 'knn', 'cv_f1_macro': kn['cv_f1_macro'], 'cv_recall_critico': None,
         'en_produccion': False, **kn['test']},
        {'modelo': 'K-Means (grupo → prioridad mayoritaria)', 'clave': 'kmeans', 'cv_f1_macro': None,
         'cv_recall_critico': None, 'en_produccion': False, **km['como_clasificador']},
    ], key=lambda r: -(r['cv_f1_macro'] or 0))
    best = ranking[0]

    result = {'fecha': datetime.now().isoformat(timespec='seconds'), 'train': len(train), 'test': len(test),
              'ranking_prioridad': ranking, 'mejor_por_cv': best['clave'], 'en_produccion': production,
              'random_forest': rf, 'logistic_regression': lr, 'knn': kn, 'linear_regression': lin,
              'kmeans': km, 'q_learning': ql}
    with open(COMPARISON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Comparación guardada en {COMPARISON_PATH}')
    return result


def load_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None
    with open(COMPARISON_PATH, encoding='utf-8') as f:
        return json.load(f)
