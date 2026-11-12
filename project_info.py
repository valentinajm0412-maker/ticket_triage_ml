"""
Contenido de la presentación pública del proyecto (Anexo 1 - Formato de Propuesta PGC).
|| Public project presentation content.

Edite aquí los textos: la página pública (/) se genera a partir de este archivo.
"""
import json
import os

from data_pipeline import ML_DIR

# ---------------------------------------------------------------- Estructura del proyecto
TITULO = 'Ticket Triage ML: priorización y asignación automática de tickets de mesa de ayuda con aprendizaje automático'
SUBTITULO = ('Sistema web que lee cada ticket, predice su prioridad con un modelo de Machine Learning y lo asigna '
             'al consultor adecuado, desarrollado con la metodología CRISP-ML.')

ENTENDIMIENTO = {
    'contexto': (
        'Una mesa de ayuda recibe a diario solicitudes de los usuarios de una organización con varias sedes y '
        'muchos sistemas: facturación, nómina, inventario, ERP, agendamiento de citas, historia clínica, VPN, entre otros. '
        'Cada solicitud (ticket) tiene un título y una descripción en texto libre, y debe ser atendida por un consultor.'),
    'actores': [
        ('Cliente', 'Usuario que reporta una falla o hace una solicitud.'),
        ('Consultor técnico', 'Atiende incidentes: algo dejó de funcionar.'),
        ('Consultor funcional', 'Atiende requerimientos: algo nuevo o un cambio.'),
        ('Administrador', 'Gestiona usuarios y supervisa la operación y el modelo.'),
    ],
    'proceso_actual': [
        'Los tickets se atienden por orden de llegada, sin una prioridad objetiva.',
        'La prioridad depende de cómo escribe el usuario: casi todos dicen "urgente".',
        'La asignación a consultores es manual, lo que genera retrasos y cargas desiguales.',
        'No queda registro de qué tan bien se priorizó, así que el proceso no mejora con el tiempo.',
    ],
    'tipos_ticket': [
        ('Incidente', 'Algo dejó de funcionar. Su gravedad depende de la severidad (caída total, error grave, falla parcial o detalle visual), del alcance (toda la empresa, un área o un usuario) y del impacto en el negocio.'),
        ('Requerimiento', 'Una solicitud nueva o un cambio: consultas, reportes, permisos o ajustes. Su urgencia depende de fechas límite como cierres, auditorías o requisitos de la DIAN.'),
    ],
}

PROBLEMA = (
    'Cuando la prioridad de un ticket se decide de forma manual y subjetiva, los casos críticos (por ejemplo, un sistema '
    'de facturación caído para toda la empresa) pueden quedar en cola detrás de consultas menores, y la carga de trabajo '
    'se reparte de forma desigual entre los consultores. Esto alarga los tiempos de respuesta en los casos que más afectan '
    'a la operación y no deja información para mejorar el proceso.')
PREGUNTA = ('¿Cómo priorizar y asignar automáticamente los tickets de una mesa de ayuda a partir de su texto, de forma que '
            'los casos críticos se atiendan primero y la carga se distribuya de manera equilibrada entre los consultores?')
JUSTIFICACION = [
    'La clasificación manual es lenta e inconsistente; la literatura reporta que automatizarla reduce el tiempo de '
    'resolución y mejora la productividad y la satisfacción de los usuarios (Ramya et al., 2021; Truss y Boehm, 2024).',
    'Dejar pasar un ticket crítico es el error más costoso para la organización; por eso el sistema prioriza detectar '
    'los críticos aunque genere algunas falsas alarmas.',
    'El texto de los tickets ya existe: no requiere capturar datos nuevos, solo aprovecharlos con técnicas de '
    'procesamiento de lenguaje natural y aprendizaje automático.',
    'Seguir una metodología formal (CRISP-ML) asegura que el modelo sea medible, reproducible, explicable y que se '
    'mantenga en el tiempo (Studer et al., 2021).',
    'El proyecto aplica en un caso real los cinco modelos del curso y deja un software funcional para la mesa de ayuda.',
]

SOLUCION = {
    'resumen': (
        'Una aplicación web en Flask donde el cliente crea el ticket y un modelo de Machine Learning, entrenado con '
        'tickets etiquetados, predice su prioridad en el momento. El sistema lo asigna al consultor del rol correcto con '
        'menos carga, marca para revisión humana los casos dudosos y aprende de las correcciones de los consultores.'),
    'pasos': [
        ('El cliente crea el ticket', 'Escribe título y descripción y elige si es Incidente o Requerimiento.'),
        ('El modelo predice la prioridad', 'Limpia el texto, lo convierte en características (TF-IDF) y calcula un puntaje de gravedad de 0 a 3: Bajo, Medio, Alto o Crítico.'),
        ('Se asigna el consultor', 'Incidentes a consultores técnicos y requerimientos a funcionales, eligiendo al de menor carga.'),
        ('El consultor confirma o corrige', 'Los tickets con baja confianza salen marcados con ⚠ para revisión humana.'),
        ('El modelo se mantiene', 'Las correcciones se exportan como datos reales; el monitoreo detecta degradación y el reentrenamiento seguro actualiza el modelo.'),
    ],
}

OBJETIVO_GENERAL = (
    'Desarrollar un sistema web de mesa de ayuda que clasifique automáticamente la prioridad de los tickets mediante '
    'aprendizaje automático y los asigne al consultor adecuado, siguiendo la metodología CRISP-ML.')
OBJETIVOS_ESPECIFICOS = [
    'Caracterizar el proceso de atención de tickets y definir los objetivos de negocio y sus indicadores: exactitud ≥ 85% y detección de críticos ≥ 95%.',
    'Construir un conjunto de datos etiquetado y un proceso reproducible de limpieza y transformación del texto.',
    'Entrenar y comparar los modelos Logistic Regression, KNN, K-Means, Linear Regression y Q-Learning, seleccionando el mejor mediante validación cruzada.',
    'Evaluar el modelo elegido con datos no vistos, analizando riesgos, sesgos y errores con revisión de expertos.',
    'Desplegar el modelo en una aplicación web con roles, asignación por carga de trabajo y retroalimentación de los consultores.',
    'Implementar el monitoreo del desempeño y de la degradación del modelo, con reentrenamiento seguro.',
]

# ---------------------------------------------------------------- Marco referencial
ANTECEDENTES = [
    ('Ramya et al. (2021)', 'Clasifican tickets no estructurados de mesa de servicio de TI combinando clasificadores con '
     'bagging, boosting y votación, y señalan como beneficios una resolución más rápida y mayor productividad.'),
    ('Paramesh y Shreedhara (2022)', 'Proponen un clasificador de tickets basado en redes neuronales convolucionales para '
     'superar problemas de los métodos tradicionales como la dispersión de los datos y el sobreajuste.'),
    ('Truss y Boehm (2024)', 'Revisan el estado del arte de la clasificación de tickets de soporte con IA e implementan '
     'una solución con AutoML para organizaciones sin experiencia técnica especializada.'),
    ('Lê y Ait Bachir (2025)', 'Comparan enfoques para priorizar tickets de TI y muestran que los modelos clásicos como '
     'TF-IDF tienen límites para captar el contexto, lo que motiva a usar ajustes de negocio sobre el modelo.'),
    ('Razma y Jurkus (2026)', 'Aplican un flujo de limpieza de texto, partición estratificada y entrenamiento supervisado '
     'para clasificar solicitudes de soporte en un sistema empresarial real con clases desbalanceadas.'),
]

MARCO_TEORICO = [
    ('CRISP-ML(Q)', 'Modelo de proceso para proyectos de Machine Learning en seis fases, desde el entendimiento del negocio '
     'hasta el monitoreo, con control de calidad en cada tarea (Studer et al., 2021).'),
    ('TF-IDF', 'Representa un texto como números: cada palabra pesa más si es frecuente en el ticket y poco común en los '
     'demás (Manning et al., 2008).'),
    ('Logistic Regression', 'Clasificador lineal que estima la probabilidad de cada clase; base sólida para texto.'),
    ('KNN', 'Clasifica un ticket según la clase más común entre sus K vecinos más parecidos.'),
    ('Linear Regression', 'Ajusta una relación lineal para predecir un valor numérico; aquí, el puntaje de gravedad 0 a 3.'),
    ('K-Means', 'Agrupa datos sin etiquetas en K grupos según su parecido (MacQueen, 1967).'),
    ('Q-Learning', 'Aprendizaje por refuerzo: aprende qué acción da más recompensa en cada situación (Watkins y Dayan, '
     '1992; Sutton y Barto, 2018).'),
    ('Validación cruzada', 'Divide el entrenamiento en partes para medir el modelo varias veces sin tocar los datos de prueba.'),
]

GLOSARIO = [
    ('Ticket', 'Solicitud registrada en la mesa de ayuda.'),
    ('Triage', 'Clasificar y ordenar los casos según su urgencia.'),
    ('Recall (Crítico)', 'De todos los tickets críticos reales, qué porcentaje detecta el modelo.'),
    ('F1 macro', 'Promedio del equilibrio entre precisión y recall de las cuatro prioridades.'),
    ('Drift', 'Cambio en los datos que llegan frente a los de entrenamiento, que degrada el modelo.'),
    ('PSI', 'Índice que mide cuánto cambió una distribución frente a la de referencia.'),
]

# ---------------------------------------------------------------- Metodología y cronograma
METODOLOGIA = (
    'Se usa CRISP-ML(Q), que extiende CRISP-DM para proyectos de Machine Learning. El proyecto es tecnológico, '
    'por lo que cada fase se materializa en código y en una hoja pública con sus resultados.')
FASES = [
    {'n': 1, 'nombre': 'Entendimiento del Negocio y los Datos', 'ruta': 'crisp_fase1', 'semanas': [1, 2],
     'actividades': 'Objetivos, KPI, preguntas clave, evaluación de datos y viabilidad'},
    {'n': 2, 'nombre': 'Ingeniería de Datos', 'ruta': 'crisp_fase2', 'semanas': [2, 3],
     'actividades': 'Generación del dataset, limpieza, transformación TF-IDF y reporte reproducible'},
    {'n': 3, 'nombre': 'Ingeniería del Modelo', 'ruta': 'crisp_fase3', 'semanas': [4, 5],
     'actividades': 'Selección de algoritmos, validación cruzada, ajuste de hiperparámetros y del corte de Crítico'},
    {'n': 4, 'nombre': 'Evaluación del Modelo', 'ruta': 'crisp_fase4', 'semanas': [6, 7],
     'actividades': 'Métricas en test, comparación de los 5 modelos, riesgos, sesgos y revisión de expertos'},
    {'n': 5, 'nombre': 'Despliegue del Modelo', 'ruta': 'crisp_fase5', 'semanas': [8, 9],
     'actividades': 'App web con roles, seguridad, retroalimentación de consultores y documentación'},
    {'n': 6, 'nombre': 'Monitoreo y Mantenimiento', 'ruta': 'crisp_fase6', 'semanas': [10, 11],
     'actividades': 'Monitoreo de desempeño y drift, reentrenamiento seguro e historial de versiones'},
]
SEMANAS = 12
HITOS = {12: 'Sustentación del PGC'}

# ---------------------------------------------------------------- Productos esperados (Anexo 1)
PRODUCTOS = [
    ('Documento síntesis de PGC', True), ('Software', True), ('Sistema de información', True), ('Prototipo', True),
    ('Artículo científico', False), ('Póster - Ponencia', False),
    ('Comunidad de práctica y redes de conocimiento', False), ('Evento científico o académico', False),
    ('Ediciones de revista o libro de divulgación', False), ('Producción de contenidos digitales', False),
    ('Publicaciones editoriales no especializadas', False), ('Desarrollo Web', True), ('Creación de cursos', False),
    ('Emprendimiento e innovación', False), ('Otro ¿Cuál?', False),
]

# ---------------------------------------------------------------- Referencias (APA)
REFERENCIAS = [
    'KDnuggets. (2025). The machine learning lifecycle. KDnuggets.',
    'Lê, M. T., y Ait Bachir, A. (2025). Learning to prioritize IT tickets: A comparative evaluation of embedding-based approaches and fine-tuned transformer models. arXiv. https://arxiv.org/abs/2512.17916',
    'MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. En Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability (Vol. 1, pp. 281-297). University of California Press.',
    'Manning, C. D., Raghavan, P., y Schütze, H. (2008). Introduction to information retrieval. Cambridge University Press.',
    'ML-Ops.org. (s.f.). CRISP-ML(Q): The ML lifecycle process. https://ml-ops.org/content/crisp-ml',
    'Paramesh, S. P., y Shreedhara, K. S. (2022). A deep learning based IT service desk ticket classifier using CNN. ICTACT Journal on Soft Computing, 13(1), 2805-2812. https://doi.org/10.21917/ijsc.2022.0399',
    'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.',
    'Ramya, C., Paramesh, S. P., y Shreedhara, K. S. (2021). Classifying the unstructured IT service desk tickets using ensemble of classifiers. arXiv. https://arxiv.org/abs/2103.15822',
    'Razma, A., y Jurkus, R. (2026). AI-based classification of IT support requests in enterprise service management systems. Systems, 14(2), 223. https://doi.org/10.3390/systems14020223',
    'Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., y Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. Machine Learning and Knowledge Extraction, 3(2), 392-413. https://arxiv.org/abs/2003.05155',
    'Sutton, R. S., y Barto, A. G. (2018). Reinforcement learning: An introduction (2.ª ed.). MIT Press.',
    'Truss, M., y Boehm, S. (2024). AI-based classification of customer support tickets: State of the art and implementation with AutoML. arXiv. https://arxiv.org/abs/2406.01789',
    'Watkins, C. J. C. H., y Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292.',
]


def _load(name):
    path = os.path.join(ML_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build_context():
    """Junta el contenido editable con los resultados reales del último entrenamiento."""
    ev, cmp, data = _load('evaluation_report.json'), _load('model_comparison.json'), _load('data_report.json')
    chosen = next((r for r in (cmp or {}).get('ranking_prioridad', []) if r.get('en_produccion')), None)
    resultados = None
    if ev:
        m = ev['metricas_test']
        resultados = {
            'modelo': chosen['modelo'] if chosen else ev.get('modelo'),
            'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'], 'recall_critico': m['recall_critico'],
            'categoria': ev.get('categoria', {}).get('accuracy'), 'decision': ev.get('decision'),
            'test': ev.get('test_filas'),
            'tickets': (data or {}).get('limpieza', {}).get('filas_finales'),
        }
    return {
        'titulo': TITULO, 'subtitulo': SUBTITULO, 'entendimiento': ENTENDIMIENTO,
        'problema': PROBLEMA, 'pregunta': PREGUNTA, 'justificacion': JUSTIFICACION, 'solucion': SOLUCION,
        'objetivo_general': OBJETIVO_GENERAL, 'objetivos_especificos': OBJETIVOS_ESPECIFICOS,
        'antecedentes': ANTECEDENTES, 'marco_teorico': MARCO_TEORICO, 'glosario': GLOSARIO,
        'metodologia': METODOLOGIA, 'fases': FASES, 'semanas': list(range(1, SEMANAS + 1)), 'hitos': HITOS,
        'productos': PRODUCTOS, 'referencias': REFERENCIAS, 'resultados': resultados,
        'ranking': (cmp or {}).get('ranking_prioridad', []),
    }
