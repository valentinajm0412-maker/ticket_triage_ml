"""
CRISP-ML Fase 3: Ingeniería del Modelo || CRISP-ML Phase 3: Model Engineering

1. Selección de algoritmos     -> CANDIDATES
2. Entrenamiento               -> Pipeline (limpieza + características + modelo) ajustado solo con train
3. Validación                  -> validación cruzada estratificada (5 folds) dentro de train
4. Ajuste de hiperparámetros   -> GridSearchCV + umbral de decisión para "Critico"

El conjunto de test NO se usa para elegir nada: queda reservado para la Fase 4 (Evaluación).
"""
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from data_pipeline import TextCleaner, build_features

SEED = 42
CV_FOLDS = 5
CRITICAL_LABEL = 'Critico'
TARGET_CRITICAL_RECALL = 0.95  # KPI definido en la Fase 1

# 1. Algoritmos candidatos y sus hiperparámetros a explorar
CANDIDATES = {
    'Naive Bayes': (ComplementNB(), {'modelo__alpha': [0.1, 0.3, 1.0]}),
    'Logistic Regression': (LogisticRegression(max_iter=3000, random_state=SEED),
                            {'modelo__C': [1, 10, 30], 'modelo__class_weight': [None, 'balanced']}),
    'Linear SVM': (LinearSVC(random_state=SEED),
                   {'modelo__C': [0.1, 0.5, 1], 'modelo__class_weight': [None, 'balanced']}),
    'Random Forest': (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
                      {'modelo__min_samples_leaf': [1, 2],
                       'modelo__class_weight': [None, 'balanced_subsample']}),
}


def make_pipeline(model):
    return Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                     ('modelo', model)])


def _critical_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[CRITICAL_LABEL], average='macro', zero_division=0)


class CriticalThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Envuelve un Pipeline y marca 'Critico' cuando su probabilidad supera un umbral.
    Bajar el umbral detecta más críticos (más recall) a costa de algunas falsas alarmas."""

    def __init__(self, pipeline=None, threshold=0.5, critical_label=CRITICAL_LABEL):
        self.pipeline = pipeline
        self.threshold = threshold
        self.critical_label = critical_label

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return apply_threshold(self.predict_proba(X), self.classes_, self.threshold, self.critical_label)


def apply_threshold(proba, classes, threshold, critical_label=CRITICAL_LABEL):
    classes = np.asarray(classes)
    crit = int(np.where(classes == critical_label)[0][0])
    others = proba.copy()
    others[:, crit] = -1
    pred = classes[others.argmax(axis=1)]
    pred[proba[:, crit] >= threshold] = critical_label
    return pred


def tune_critical_threshold(pipeline, X, y, cv):
    """Busca con predicciones fuera de fold el umbral que cumple el recall objetivo
    con el mejor F1 macro posible."""
    proba = cross_val_predict(clone(pipeline), X, y, cv=cv, method='predict_proba')
    classes = np.unique(y)
    curve = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = apply_threshold(proba, classes, t)
        curve.append({'umbral': float(t), 'recall_critico': round(_critical_recall(y, pred), 4),
                      'f1_macro': round(f1_score(y, pred, average='macro'), 4)})
    default = {'recall_critico': round(_critical_recall(y, classes[proba.argmax(axis=1)]), 4),
               'f1_macro': round(f1_score(y, classes[proba.argmax(axis=1)], average='macro'), 4)}
    meets = [c for c in curve if c['recall_critico'] >= TARGET_CRITICAL_RECALL]
    best = max(meets, key=lambda c: c['f1_macro']) if meets else max(
        curve, key=lambda c: (c['recall_critico'], c['f1_macro']))
    return best, default, bool(meets), curve[::5]


# ---------------------------------------------------------------- Regresión lineal ordinal
SEVERITY_SCORE = {'Bajo': 0, 'Medio': 1, 'Alto': 2, 'Critico': 3}
SCORE_LABEL = {v: k for k, v in SEVERITY_SCORE.items()}


def make_ordinal_pipeline():
    """Texto -> limpieza -> TF-IDF -> SVD (100 dimensiones) -> regresión lineal sobre el puntaje 0-3."""
    return Pipeline([('limpieza', TextCleaner()), ('caracteristicas', build_features()),
                     ('svd', TruncatedSVD(n_components=100, random_state=SEED)), ('norma', Normalizer()),
                     ('modelo', LinearRegression())])


def scores_to_labels(scores, critical_cut=2.5):
    """Puntaje >= corte -> Crítico; el resto se redondea a Bajo/Medio/Alto."""
    return np.array([CRITICAL_LABEL if v >= critical_cut else SCORE_LABEL[int(np.clip(np.rint(v), 0, 2))]
                     for v in scores])


class OrdinalLinearClassifier(BaseEstimator, ClassifierMixin):
    """La prioridad es un orden (Bajo < Medio < Alto < Crítico): se predice un puntaje 0-3 con regresión
    lineal y se convierte en etiqueta. predict_proba da una confianza aproximada según qué tan cerca
    está el puntaje de cada prioridad (usa el error típico medido en validación cruzada)."""

    def __init__(self, critical_cut=2.5, sigma=0.5):
        self.critical_cut = critical_cut
        self.sigma = sigma

    def fit(self, X, y):
        self.pipeline = make_ordinal_pipeline().fit(X, [SEVERITY_SCORE[v] for v in y])
        self.classes_ = np.array(sorted(SEVERITY_SCORE))  # orden alfabético, como los demás modelos
        return self

    def score_ticket(self, X):
        return self.pipeline.predict(X)

    def predict(self, X):
        return scores_to_labels(self.score_ticket(X), self.critical_cut)

    def predict_proba(self, X):
        scores = self.score_ticket(X)
        centers = np.array([SEVERITY_SCORE[c] for c in self.classes_], dtype=float)
        # El centro de Crítico se corre hasta el corte para que la confianza sea coherente con predict
        centers[list(self.classes_).index(CRITICAL_LABEL)] = max(self.critical_cut + 0.5, 2.5)
        logits = -((scores[:, None] - centers[None, :]) ** 2) / (2 * self.sigma ** 2)
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        return proba / proba.sum(axis=1, keepdims=True)


def evaluate_ordinal(X, y, cv):
    """Validación cruzada del modelo ordinal y ajuste del corte de Crítico con predicciones fuera de fold."""
    start = time.time()
    y_score = np.array([SEVERITY_SCORE[v] for v in y])
    folds = list(cv.split(X, y))  # mismas particiones que los demás algoritmos
    oof = cross_val_predict(make_ordinal_pipeline(), X, y_score, cv=folds)
    y_arr = np.asarray(y)

    def fold_metrics(cut):
        f1s, accs, recs = [], [], []
        for _, te in folds:
            pred = scores_to_labels(oof[te], cut)
            f1s.append(f1_score(y_arr[te], pred, average='macro'))
            accs.append(np.mean(pred == y_arr[te]))
            recs.append(_critical_recall(y_arr[te], pred))
        return {'f1_macro': round(float(np.mean(f1s)), 4), 'f1_macro_std': round(float(np.std(f1s)), 4),
                'accuracy': round(float(np.mean(accs)), 4), 'recall_critico': round(float(np.mean(recs)), 4)}

    default = fold_metrics(2.5)
    best_cut, best = 2.5, default
    for cut in np.round(np.arange(1.6, 2.6, 0.02), 2):
        m = fold_metrics(cut)
        if m['recall_critico'] >= TARGET_CRITICAL_RECALL and (
                best['recall_critico'] < TARGET_CRITICAL_RECALL or m['f1_macro'] > best['f1_macro']):
            best_cut, best = float(cut), m
    sigma = float(np.std(oof - y_score)) or 0.5
    return {'corte': best_cut, 'sigma': round(sigma, 4), 'antes': default, 'despues': best,
            'cumple': best['recall_critico'] >= TARGET_CRITICAL_RECALL, 'segundos': round(time.time() - start, 1)}


def select_and_train(target_name, X_train, y_train, business_rule=False):
    """Compara algoritmos con GridSearchCV y devuelve (modelo_final, resumen)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}
    if business_rule:
        scoring['recall_critico'] = make_scorer(_critical_recall)

    print(f'\n=== {target_name.upper()}: validación cruzada {CV_FOLDS} folds sobre train ===')
    results, searches = [], {}
    for name, (model, grid) in CANDIDATES.items():
        start = time.time()
        search = GridSearchCV(make_pipeline(model), grid, scoring=scoring, refit='f1_macro',
                              cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        i = search.best_index_
        cvr = search.cv_results_
        row = {
            'algoritmo': name,
            'mejores_parametros': {k.replace('modelo__', ''): str(v) for k, v in search.best_params_.items()},
            'combinaciones_probadas': len(cvr['params']),
            'f1_macro': round(cvr['mean_test_f1_macro'][i], 4),
            'f1_macro_std': round(cvr['std_test_f1_macro'][i], 4),
            'accuracy': round(cvr['mean_test_accuracy'][i], 4),
            'segundos': round(time.time() - start, 1),
            'soporta_probabilidad': hasattr(search.best_estimator_, 'predict_proba'),
        }
        if business_rule:
            row['recall_critico'] = round(cvr['mean_test_recall_critico'][i], 4)
        results.append(row)
        searches[name] = search
        print(f"{name:20} F1={row['f1_macro']:.4f} (±{row['f1_macro_std']:.4f}) "
              f"Acc={row['accuracy']:.4f}"
              + (f" RecallCrit={row['recall_critico']:.4f}" if business_rule else '')
              + f"  {row['mejores_parametros']}")

    best_row = max(results, key=lambda r: r['f1_macro'])
    summary = {'candidatos': results, 'seleccionado': best_row['algoritmo'],
               'criterio': 'Mayor F1 macro promedio en validación cruzada (no se usa el test)'}
    final_model = searches[best_row['algoritmo']].best_estimator_

    if business_rule:
        # Entre los modelos con probabilidad, el de mejor F1; luego se ajusta el umbral de críticos
        prob_rows = [r for r in results if r['soporta_probabilidad']]
        best_prob = max(prob_rows, key=lambda r: r['f1_macro'])
        pipe = searches[best_prob['algoritmo']].best_estimator_
        chosen, default, met, curve = tune_critical_threshold(pipe, X_train, y_train, cv)
        print(f"Umbral Critico para {best_prob['algoritmo']}: {chosen['umbral']} -> "
              f"recall {default['recall_critico']} -> {chosen['recall_critico']}, "
              f"F1 {default['f1_macro']} -> {chosen['f1_macro']}")
        summary['umbral'] = {'algoritmo': best_prob['algoritmo'], 'objetivo_recall': TARGET_CRITICAL_RECALL,
                             'umbral': chosen['umbral'], 'cumple': met,
                             'antes': default, 'despues': {k: chosen[k] for k in ('recall_critico', 'f1_macro')},
                             'curva': curve}
        criterio = (f'Cumplir recall en Critico ≥ {TARGET_CRITICAL_RECALL:.0%} (KPI Fase 1) '
                    'con el mejor F1 macro posible, medido en validación cruzada')
        if met or chosen['recall_critico'] > best_row.get('recall_critico', 0):
            summary['seleccionado'] = f"{best_prob['algoritmo']} + umbral Critico {chosen['umbral']}"
            summary['criterio'] = criterio
            summary['cv_seleccionado'] = {'f1_macro': chosen['f1_macro'], 'recall_critico': chosen['recall_critico'],
                                          'f1_macro_std': best_prob['f1_macro_std'], 'accuracy': best_prob['accuracy']}
            final_model = CriticalThresholdClassifier(pipe, threshold=chosen['umbral']).fit(X_train, y_train)

        # Prioridad como puntaje ordenado 0-3 (regresión lineal) con corte para Crítico
        ordn = evaluate_ordinal(X_train, y_train, cv)
        name = 'Linear Regression (ordinal)'
        results.append({'algoritmo': name, 'mejores_parametros': {'corte_critico': str(ordn['corte']),
                                                                  'svd': '100'},
                        'combinaciones_probadas': 50, 'f1_macro': ordn['despues']['f1_macro'],
                        'f1_macro_std': ordn['despues']['f1_macro_std'], 'accuracy': ordn['despues']['accuracy'],
                        'recall_critico': ordn['despues']['recall_critico'], 'segundos': ordn['segundos'],
                        'soporta_probabilidad': True})
        summary['ordinal'] = ordn
        print(f"{name:20} F1={ordn['despues']['f1_macro']:.4f} (±{ordn['despues']['f1_macro_std']:.4f}) "
              f"Acc={ordn['despues']['accuracy']:.4f} RecallCrit={ordn['despues']['recall_critico']:.4f}"
              f"  corte Critico {ordn['corte']}")
        current_f1 = summary.get('cv_seleccionado', {}).get('f1_macro', best_row['f1_macro'])
        if ordn['cumple'] and ordn['despues']['f1_macro'] > current_f1:
            summary['seleccionado'] = f"{name} + corte Critico {ordn['corte']}"
            summary['criterio'] = criterio
            summary['cv_seleccionado'] = ordn['despues']
            final_model = OrdinalLinearClassifier(ordn['corte'], ordn['sigma']).fit(X_train, y_train)

    print(f"Seleccionado: {summary['seleccionado']}")
    return final_model, summary
