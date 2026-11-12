"""
CRISP-ML Fase 1: Entendimiento del Negocio y los Datos
|| CRISP-ML Phase 1: Business and Data Understanding

Contenido de negocio (objetivos, preguntas, viabilidad) + evaluación
automática del dataset de entrenamiento y de los tickets reales en BD.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = os.path.join('ml_models', 'ticket_dataset.csv')

# 1. Objetivos de negocio || Business objectives
BUSINESS_OBJECTIVES = [
    {
        'necesidad': 'Los tickets llegan sin priorizar y se atienden por orden de llegada.',
        'meta': 'Asignar automáticamente la prioridad (Crítico, Alto, Medio, Bajo) al crear el ticket.',
        'kpi': 'Exactitud ≥ 85% y recall en "Crítico" ≥ 95% sobre datos reales etiquetados.',
    },
    {
        'necesidad': 'La asignación manual de consultores genera retrasos y cargas desiguales.',
        'meta': 'Enrutar cada ticket al tipo de consultor correcto con el menor workload.',
        'kpi': 'Diferencia máxima de carga entre consultores del mismo rol ≤ 2 tickets.',
    },
    {
        'necesidad': 'Los incidentes críticos tardan en ser atendidos.',
        'meta': 'Reducir el tiempo de primera respuesta de los tickets críticos.',
        'kpi': 'Tiempo de "Abierto" a "En Proceso" en críticos < 30 min (línea base por medir).',
    },
]

# 2. Preguntas clave || Key questions
KEY_QUESTIONS = [
    '¿Qué prioridad tiene un ticket a partir de su título y descripción?',
    '¿Es un Incidente (consultor técnico) o un Requerimiento (consultor funcional)?',
    '¿Qué consultor disponible debe atenderlo para balancear la carga?',
    '¿Qué tan seguro está el modelo de su predicción (para pedir revisión humana si es baja)?',
]

# 4. Análisis de viabilidad || Feasibility analysis
FEASIBILITY = [
    {'aspecto': 'Técnico', 'estado': 'Viable',
     'detalle': 'TF-IDF + clasificadores de scikit-learn son ligeros y se integran ya en Flask.'},
    {'aspecto': 'Datos', 'estado': 'Riesgo alto',
     'detalle': 'El dataset es sintético y poco variado; se necesitan tickets reales etiquetados.'},
    {'aspecto': 'Organizacional', 'estado': 'Viable con condiciones',
     'detalle': 'Requiere que los consultores validen/corrijan la prioridad para retroalimentar el modelo.'},
    {'aspecto': 'Riesgo de negocio', 'estado': 'Mitigable',
     'detalle': 'Un crítico clasificado como bajo es costoso: priorizar recall en "Crítico" y permitir ajuste manual.'},
    {'aspecto': 'Privacidad', 'estado': 'Por revisar',
     'detalle': 'Las descripciones pueden incluir datos de clientes; definir anonimización y retención.'},
]


def evaluate_dataset(path=DATASET_PATH):
    """3. Evaluación de datos: calidad, relevancia y suficiencia."""
    if not os.path.exists(path):
        return {'error': f'No se encontró el dataset en {path}'}

    df = pd.read_csv(path)
    n = len(df)
    text = df['descripcion'].astype(str)
    unique_desc = text.nunique()
    words = text.str.split()
    vocab = set(w.lower() for ws in words for w in ws)

    # Solapamiento train/test con el mismo split de train_models.py
    _, x_test, _, _ = train_test_split(
        text, df['prioridad'], test_size=0.2, random_state=42, stratify=df['prioridad'])
    x_train = set(text.drop(x_test.index))
    overlap_pct = round(100 * x_test.isin(x_train).mean(), 1)

    priority_counts = df['prioridad'].value_counts()
    crosstab = pd.crosstab(df['prioridad'], df['categoria'])

    checks = [
        ('Valores nulos', int(df.isna().sum().sum()) == 0,
         f"{int(df.isna().sum().sum())} nulos"),
        ('Variedad de textos (≥ 50% únicos)', unique_desc / n >= 0.5,
         f"{unique_desc} descripciones únicas de {n} ({100 * unique_desc / n:.1f}%)"),
        ('Test independiente del train (solapamiento < 5%)', overlap_pct < 5,
         f"{overlap_pct}% de los textos de prueba también están en entrenamiento"),
        ('Balance de clases (mín/máx ≥ 0.3)', priority_counts.min() / priority_counts.max() >= 0.3,
         f"Clase menor {priority_counts.idxmin()} = {priority_counts.min()}, "
         f"mayor {priority_counts.idxmax()} = {priority_counts.max()}"),
        ('Suficiencia (≥ 100 ejemplos únicos por clase)',
         df.drop_duplicates('descripcion').groupby('prioridad').size().min() >= 100,
         'Ejemplos únicos por clase: ' + ', '.join(
             f"{k}={v}" for k, v in df.drop_duplicates('descripcion').groupby('prioridad').size().items())),
        ('Origen de datos real', False,
         'Dataset sintético generado por plantillas (generate_dataset.py)'),
    ]

    return {
        'rows': n,
        'columns': list(df.columns),
        'unique_desc': unique_desc,
        'duplicate_pct': round(100 * (1 - unique_desc / n), 1),
        'avg_words': round(words.str.len().mean(), 1),
        'vocab_size': len(vocab),
        'overlap_pct': overlap_pct,
        'priority_dist': [(k, int(v), round(100 * v / n, 1)) for k, v in priority_counts.items()],
        'crosstab_cols': list(crosstab.columns),
        'crosstab': [(idx, [int(v) for v in row]) for idx, row in crosstab.iterrows()],
        'checks': [{'nombre': c, 'ok': bool(ok), 'detalle': d} for c, ok, d in checks],
    }


def evaluate_production_tickets(ticket_model):
    """Resumen de los tickets reales registrados en la aplicación."""
    tickets = ticket_model.query.all()
    if not tickets:
        return {'total': 0}
    df = pd.DataFrame([{'prioridad': t.priority, 'categoria': t.category,
                        'estado': t.status} for t in tickets])
    return {
        'total': len(df),
        'por_prioridad': df['prioridad'].fillna('Sin prioridad').value_counts().to_dict(),
        'por_categoria': df['categoria'].value_counts().to_dict(),
        'por_estado': df['estado'].value_counts().to_dict(),
    }


def build_context(ticket_model):
    return {
        'objectives': BUSINESS_OBJECTIVES,
        'questions': KEY_QUESTIONS,
        'data_eval': evaluate_dataset(),
        'prod': evaluate_production_tickets(ticket_model),
        'feasibility': FEASIBILITY,
    }
