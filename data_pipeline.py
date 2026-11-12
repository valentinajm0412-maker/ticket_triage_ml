"""
CRISP-ML Fase 2: Ingeniería de Datos || CRISP-ML Phase 2: Data Engineering

1. Selección      -> load_sources()
2. Limpieza       -> clean_dataset()
3. Transformación -> TextCleaner + build_features() (se usan dentro del Pipeline)
4. Documentación  -> save_report() escribe ml_models/data_report.json
"""
import hashlib
import json
import os
import platform
import re
import unicodedata
from datetime import datetime

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

ML_DIR = 'ml_models'
SYNTHETIC_PATH = os.path.join(ML_DIR, 'ticket_dataset.csv')
REAL_PATH = os.path.join(ML_DIR, 'real_tickets.csv')        # opcional: tickets reales etiquetados
CLEAN_PATH = os.path.join(ML_DIR, 'ticket_dataset_clean.csv')
REPORT_PATH = os.path.join(ML_DIR, 'data_report.json')

REQUIRED_COLUMNS = ['titulo', 'descripcion', 'categoria', 'prioridad']
VALID_PRIORITIES = ['Critico', 'Alto', 'Medio', 'Bajo']
VALID_CATEGORIES = ['Incidente', 'Requerimiento']
MIN_WORDS = 3

# Saludos y cierres que no aportan información || Greetings/closings with no signal
NOISE_PATTERNS = [
    r'(^|[.!?]\s*)(buenos dias|buenas tardes|buenas noches|hola equipo|hola|por favor ayuda)\b[,:]?\s*',
    r'\s*(gracias|quedo atento|quedo atenta|adjunto pantallazo|saludos)\.?\s*$',
]

# Stopwords en español SIN tildes. No incluye 'no', 'sin', 'todo(s)', 'ningun'
# porque cambian el significado (ej. "no responde", "todos los usuarios").
STOPWORDS_ES = [
    'a', 'al', 'algo', 'ante', 'con', 'como', 'de', 'del', 'desde', 'donde', 'e', 'el', 'en',
    'entre', 'es', 'esa', 'ese', 'eso', 'esta', 'este', 'esto', 'ha', 'hay', 'la', 'las', 'le',
    'les', 'lo', 'los', 'me', 'mi', 'mis', 'muy', 'nos', 'o', 'para', 'pero', 'por', 'que',
    'se', 'si', 'su', 'sus', 'tambien', 'te', 'tu', 'un', 'una', 'uno', 'unos', 'unas', 'y', 'ya',
]

LABEL_ALIASES = {  # normalización de etiquetas escritas distinto
    'critico': 'Critico', 'crítico': 'Critico', 'alto': 'Alto', 'medio': 'Medio', 'bajo': 'Bajo',
    'incidente': 'Incidente', 'requerimiento': 'Requerimiento',
}


# ---------------------------------------------------------------- 1. Selección
def load_sources():
    """Carga y une todas las fuentes de datos disponibles."""
    frames, sources = [], []
    for name, path in [('sintetico', SYNTHETIC_PATH), ('real', REAL_PATH)]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['fuente'] = name
            frames.append(df)
            sources.append({'fuente': name, 'archivo': path, 'filas': len(df),
                            'sha256': file_hash(path)})
        else:
            sources.append({'fuente': name, 'archivo': path, 'filas': 0, 'sha256': None})
    if not frames:
        raise FileNotFoundError(f'No hay datos. Ejecuta generate_dataset.py ({SYNTHETIC_PATH})')
    return pd.concat(frames, ignore_index=True), sources


# ---------------------------------------------------------------- 2. Limpieza
def normalize_text(text):
    """Minúsculas, sin tildes, sin saludos/cierres, sin signos, espacios simples."""
    text = str(text).lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, r'\1' if pattern.startswith('(^') else '', text)
    text = re.sub(r'[^a-z0-9ñ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def clean_dataset(df):
    """Aplica las reglas de limpieza y devuelve (df_limpio, bitácora de pasos)."""
    log = []

    def step(name, before, df_after, detalle=''):
        log.append({'paso': name, 'eliminadas': before - len(df_after),
                    'filas_resultantes': len(df_after), 'detalle': detalle})
        return df_after

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if 'titulo' in missing:  # datasets antiguos sin título
        df['titulo'] = ''
        missing.remove('titulo')
    if missing:
        raise ValueError(f'Faltan columnas obligatorias: {missing}')

    n = len(df)
    df = step('Filas cargadas', n, df, 'Unión de todas las fuentes')

    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna('').astype(str).str.strip()

    n = len(df)
    df = step('Nulos o vacíos', n, df[(df['descripcion'] != '') & (df['prioridad'] != '')
                                     & (df['categoria'] != '')],
              'Sin descripción, prioridad o categoría')

    df['prioridad'] = df['prioridad'].str.lower().map(LABEL_ALIASES).fillna(df['prioridad'])
    df['categoria'] = df['categoria'].str.lower().map(LABEL_ALIASES).fillna(df['categoria'])
    n = len(df)
    df = step('Etiquetas inválidas', n, df[df['prioridad'].isin(VALID_PRIORITIES)
                                           & df['categoria'].isin(VALID_CATEGORIES)],
              f'Prioridad fuera de {VALID_PRIORITIES} o categoría fuera de {VALID_CATEGORIES}')

    df['texto'] = df['titulo'] + '. ' + df['descripcion']
    df['texto_limpio'] = df['texto'].map(normalize_text)

    n = len(df)
    df = step('Texto muy corto', n, df[df['texto_limpio'].str.split().str.len() >= MIN_WORDS],
              f'Menos de {MIN_WORDS} palabras después de limpiar')

    n = len(df)
    df = step('Duplicados', n, df.drop_duplicates(subset=['texto_limpio', 'prioridad', 'categoria']),
              'Mismo texto normalizado y mismas etiquetas')

    conflicts = df.groupby('texto_limpio')['prioridad'].transform('nunique') > 1
    n = len(df)
    df = step('Etiquetas contradictorias', n, df[~conflicts],
              'Mismo texto con prioridades distintas (no se puede saber cuál es correcta)')

    return df.reset_index(drop=True), log


# ---------------------------------------------------------------- 3. Transformación
class TextCleaner(BaseEstimator, TransformerMixin):
    """Paso del Pipeline: aplica normalize_text igual en entrenamiento y en la app."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_text(t) for t in X]


def build_features():
    """Características: n-gramas de palabras + n-gramas de caracteres (tolera errores de digitación)."""
    return FeatureUnion([
        ('palabras', TfidfVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS_ES,
                                     min_df=2, sublinear_tf=True)),
        ('caracteres', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                       min_df=3, sublinear_tf=True, max_features=20000)),
    ])


# ---------------------------------------------------------------- 4. Documentación
def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def environment_info():
    return {'python': platform.python_version(), 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__}


def prepare_dataset():
    """Selección + limpieza. Guarda el dataset limpio y devuelve (df, reporte)."""
    raw, sources = load_sources()
    clean, log = clean_dataset(raw)
    clean[['titulo', 'descripcion', 'texto_limpio', 'categoria', 'prioridad', 'fuente']].to_csv(
        CLEAN_PATH, index=False)

    report = {
        'fecha': datetime.now().isoformat(timespec='seconds'),
        'entorno': environment_info(),
        'seleccion': {'fuentes': sources, 'columnas_requeridas': REQUIRED_COLUMNS},
        'limpieza': {'pasos': log, 'filas_iniciales': len(raw), 'filas_finales': len(clean),
                     'dataset_limpio': CLEAN_PATH, 'sha256_limpio': file_hash(CLEAN_PATH)},
        'transformacion': {
            'entrada': 'titulo + ". " + descripcion',
            'normalizacion': ['minúsculas', 'sin tildes', 'sin saludos/cierres',
                              'sin signos de puntuación', 'espacios simples'],
            'caracteristicas': ['TF-IDF palabras (1-2 gramas, stopwords español, min_df=2)',
                                'TF-IDF caracteres (3-5 gramas, min_df=3, máx. 20.000)'],
            'stopwords': len(STOPWORDS_ES),
            'ejemplos': [{'original': r.texto, 'limpio': r.texto_limpio}
                         for r in clean.sample(min(5, len(clean)), random_state=1).itertuples()],
        },
        'distribucion': {
            'prioridad': clean['prioridad'].value_counts().to_dict(),
            'categoria': clean['categoria'].value_counts().to_dict(),
            'fuente': clean['fuente'].value_counts().to_dict(),
        },
    }
    return clean, report


def save_report(report, path=REPORT_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_report(path=REPORT_PATH):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------- Fase 6: apoyo al monitoreo
def word_vocabulary(model):
    """Vocabulario de palabras que aprendió el modelo (para detectar palabras nuevas)."""
    pipe = getattr(model, 'pipeline', model)
    feats = pipe.named_steps['caracteristicas']
    vectorizer = dict(feats.transformer_list)['palabras']
    return {term for term in vectorizer.vocabulary_ if ' ' not in term}


def oov_stats(texts, vocabulary, top=10):
    """% de palabras desconocidas para el modelo y las más frecuentes (fuera de vocabulario)."""
    stop = set(STOPWORDS_ES)
    total, unknown, counts = 0, 0, {}
    for t in texts:
        for w in normalize_text(t).split():
            if w in stop or len(w) < 3 or w.isdigit():
                continue
            total += 1
            if w not in vocabulary:
                unknown += 1
                counts[w] = counts.get(w, 0) + 1
    top_words = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return {'tasa': round(unknown / total, 4) if total else 0.0, 'palabras': total, 'top': top_words}
