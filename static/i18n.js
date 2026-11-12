// High-Speed Bilingual Engine (EN default / ES toggle) || Motor Bilingüe de Alta Velocidad (EN por defecto / cambio a ES)
(function () {
    const CACHE_KEY = 'tt_i18n_dynamic_cache_v2';
    let currentLang = 'en';
    let dynamicCache = {};

    try {
        dynamicCache = JSON.parse(localStorage.getItem(CACHE_KEY) || '{}');
    } catch (e) {
        dynamicCache = {};
    }

    // Exact phrase dictionary (Spanish -> English) || Diccionario de frases exactas (Español -> Inglés)
    const exactDict = {
        // Navigation & Global || Navegación y Global
        "Proyecto de Gestión del Conocimiento": "Knowledge Management Project",
        "El proyecto": "The project",
        "Ingresar a la aplicación": "Log in to app",
        "Ir a la aplicación": "Go to app",
        "Sitio del proyecto": "Project site",
        "Gestionar Usuarios": "Manage Users",
        "Crear Ticket": "Create Ticket",
        "Salir": "Log out",
        "1 Entendimiento del Negocio y los Datos": "1 Business & Data Understanding",
        "2 Ingeniería de Datos": "2 Data Engineering",
        "3 Ingeniería del Modelo": "3 Model Engineering",
        "4 Evaluación del Modelo": "4 Model Evaluation",
        "5 Despliegue del Modelo": "5 Model Deployment",
        "6 Monitoreo y Mantenimiento": "6 Monitoring & Maintenance",
        "1 · Entendimiento del Negocio y los Datos": "1 · Business & Data Understanding",
        "4 · Evaluación del Modelo": "4 · Model Evaluation",
        "Entendimiento del Negocio y los Datos": "Business & Data Understanding",
        "Ingeniería de Datos": "Data Engineering",
        "Ingeniería del Modelo": "Model Engineering",
        "Evaluación del Modelo": "Model Evaluation",
        "Despliegue del Modelo": "Model Deployment",
        "Monitoreo y Mantenimiento": "Monitoring & Maintenance",
        "Referencias": "References",

        // Login || Inicio de Sesión
        "Ingreso a la aplicación": "Application Login",
        "Ingresar": "Log in",
        "← Volver a la presentación del proyecto": "← Back to project presentation",

        // Dashboard & Tickets || Tablero y Tickets
        "Panel Principal": "Main Dashboard",
        "Listado de Tickets": "Ticket List",
        "Total tickets": "Total tickets",
        "Abiertos": "Open",
        "Cerrados": "Closed",
        "Altos": "High",
        "Medios": "Medium",
        "Bajos": "Low",
        "Sin datos": "No data",
        "ID": "ID",
        "Título": "Title",
        "Categoría": "Category",
        "Prioridad (ML)": "Priority (ML)",
        "Asignado A": "Assigned To",
        "Estado": "Status",
        "Revisar prioridad": "Review priority",
        "Acción": "Action",
        "Sin clasificar": "Unclassified",
        "Sin asignar": "Unassigned",
        "Actualizar": "Update",
        "Confirmar": "Confirm",
        "[Tomar caso]": "[Take case]",
        "[Resolver]": "[Resolve]",
        "Completado": "Completed",
        "No hay tickets registrados en esta vista.": "No tickets registered in this view.",
        "Abierto": "Open",
        "En Proceso": "In Progress",
        "Resuelto": "Resolved",
        "Critico": "Critical",
        "Crítico": "Critical",
        "Alto": "High",
        "Medio": "Medium",
        "Bajo": "Low",
        "Incidente": "Incident",
        "Requerimiento": "Requirement",

        // Create Ticket & Admin Users || Crear Ticket y Gestión de Usuarios
        "Crear Nuevo Ticket": "Create New Ticket",
        "Título del caso:": "Case title:",
        "Tipo de caso:": "Case type:",
        "Incidente (Soporte Técnico)": "Incident (Technical Support)",
        "Requerimiento (Soporte Funcional)": "Requirement (Functional Support)",
        "Descripción detallada:": "Detailed description:",
        "Enviar Caso": "Submit Case",
        "Gestión de Usuarios": "User Management",
        "Usuario": "User",
        "Rol": "Role",
        "Carga Actual": "Current Workload",
        "Restablecer Clave": "Reset Password",

        // Index Hero & TOC || Portada e Índice
        "Ver la solución": "View solution",
        "Recorrer las fases CRISP-ML": "Explore CRISP-ML phases",
        "exactitud en prioridad": "priority accuracy",
        "tickets críticos detectados": "critical tickets detected",
        "exactitud en categoría": "category accuracy",
        "tickets para entrenar y evaluar": "tickets for training & evaluation",
        "fases CRISP-ML implementadas": "CRISP-ML phases implemented",
        "Entendimiento": "Understanding",
        "Problema": "Problem",
        "Solución": "Solution",
        "Objetivos": "Objectives",
        "Marco referencial": "Reference Framework",
        "Metodología y cronograma": "Methodology & schedule",
        "Productos": "Deliverables",

        // Index Section 1 || Sección 1
        "Entendimiento del problema": "Problem Understanding",
        "¿Quiénes participan?": "Who participates?",
        "¿Cómo funciona hoy?": "How does it work today?",
        "Tipos de ticket": "Ticket types",
        "Cliente": "Client",
        "Consultor técnico": "Technical consultant",
        "Consultor funcional": "Functional consultant",
        "Administrador": "Administrator",
        "Algo dejó de funcionar. Su gravedad depende de la severidad (caída total, error grave, falla parcial o detalle visual), del alcance (toda la empresa, un área o un usuario) y del impacto en el negocio.": "Something stopped working. Its severity depends on impact level (total outage, critical error, partial failure, or visual glitch), scope (entire company, a department, or a single user), and business impact.",
        "Una solicitud nueva o un cambio: consultas, reportes, permisos o ajustes. Su urgencia depende de fechas límite como cierres, auditorías o requisitos de la DIAN.": "A new request or change: inquiries, reports, permissions, or adjustments. Its urgency depends on deadlines such as financial closings, audits, or DIAN regulatory requirements.",
        "El detalle de los objetivos de negocio, indicadores y evaluación de los datos está en la fase": "Details on business objectives, indicators, and data evaluation are in phase",

        // Index Section 2 || Sección 2
        "Planteamiento del problema y justificación": "Problem Statement & Justification",
        "Cuando la prioridad de un ticket se decide de forma manual y subjetiva, los casos críticos (por ejemplo, un sistema de facturación caído para toda la empresa) pueden quedar en cola detrás de consultas menores, y la carga de trabajo se reparte de forma desigual entre los consultores. Esto alarga los tiempos de respuesta en los casos que más afectan a la operación y no deja información para mejorar el proceso.": "When ticket priority is decided manually and subjectively, critical cases (for example, a company-wide billing system outage) can get stuck in the queue behind minor inquiries, and workload is unevenly distributed among consultants. This lengthens response times for the cases that affect operations the most and leaves no data to improve the process.",
        "¿Cómo priorizar y asignar automáticamente los tickets de una mesa de ayuda a partir de su texto, de forma que los casos críticos se atiendan primero y la carga se distribuya de manera equilibrada entre los consultores?": "How can helpdesk tickets be automatically prioritized and assigned based on their text, so that critical cases are handled first and workload is evenly distributed among consultants?",
        "Justificación": "Justification",
        "La clasificación manual es lenta e inconsistente; la literatura reporta que automatizarla reduce el tiempo de resolución y mejora la productividad y la satisfacción de los usuarios (Ramya et al., 2021; Truss y Boehm, 2024).": "Manual classification is slow and inconsistent; literature reports that automating it reduces resolution time and improves productivity and user satisfaction (Ramya et al., 2021; Truss & Boehm, 2024).",
        "Dejar pasar un ticket crítico es el error más costoso para la organización; por eso el sistema prioriza detectar los críticos aunque genere algunas falsas alarmas.": "Missing a critical ticket is the costliest error for the organization; therefore, the system prioritizes detecting critical cases even if it generates some false alarms.",
        "El texto de los tickets ya existe: no requiere capturar datos nuevos, solo aprovecharlos con técnicas de procesamiento de lenguaje natural y aprendizaje automático.": "Ticket text already exists: it does not require capturing new data, only leveraging it with natural language processing and machine learning techniques.",
        "Seguir una metodología formal (CRISP-ML) asegura que el modelo sea medible, reproducible, explicable y que se mantenga en el tiempo (Studer et al., 2021).": "Following a formal methodology (CRISP-ML) ensures the model is measurable, reproducible, explainable, and maintainable over time (Studer et al., 2021).",
        "El proyecto aplica en un caso real los cinco modelos del curso y deja un software funcional para la mesa de ayuda.": "The project applies the five course models to a real case and delivers functional software for the helpdesk.",

        // Index Section 3 || Sección 3
        "Una aplicación web en Flask donde el cliente crea el ticket y un modelo de Machine Learning, entrenado con tickets etiquetados, predice su prioridad en el momento. El sistema lo asigna al consultor del rol correcto con menos carga, marca para revisión humana los casos dudosos y aprende de las correcciones de los consultores.": "A Flask web application where the client creates a ticket and a Machine Learning model, trained on labeled tickets, predicts its priority in real time. The system assigns it to the consultant with the right role and lowest workload, flags uncertain cases for human review, and learns from consultant corrections.",
        "El cliente crea el ticket": "Client creates the ticket",
        "Escribe título y descripción y elige si es Incidente o Requerimiento.": "Enters title and description and selects whether it is an Incident or Requirement.",
        "El modelo predice la prioridad": "Model predicts priority",
        "Limpia el texto, lo convierte en características (TF-IDF) y calcula un puntaje de gravedad de 0 a 3: Bajo, Medio, Alto o Crítico.": "Cleans text, converts it into features (TF-IDF), and computes a severity score from 0 to 3: Low, Medium, High, or Critical.",
        "Se asigna el consultor": "Consultant is assigned",
        "Incidentes a consultores técnicos y requerimientos a funcionales, eligiendo al de menor carga.": "Incidents go to technical consultants and requirements to functional consultants, selecting the one with the lowest workload.",
        "El consultor confirma o corrige": "Consultant confirms or corrects",
        "Los tickets con baja confianza salen marcados con ⚠ para revisión humana.": "Low-confidence tickets are flagged with ⚠ for human review.",
        "El modelo se mantiene": "Model is maintained",
        "Las correcciones se exportan como datos reales; el monitoreo detecta degradación y el reentrenamiento seguro actualiza el modelo.": "Corrections are exported as real data; monitoring detects drift and safe retraining updates the model.",
        "Modelo elegido y resultados": "Selected model & results",
        "Modelo comparado": "Compared model",
        "F1 macro (validación cruzada)": "Macro F1 (cross-validation)",
        "Exactitud (test)": "Accuracy (test)",
        "Críticos detectados (test)": "Critical detected (test)",
        "K-Means (grupo → prioridad mayoritaria)": "K-Means (cluster → majority priority)",
        "Apto con condiciones: revisión humana de tickets de baja confianza y monitoreo semanal.": "Approved with conditions: human review of low-confidence tickets and weekly monitoring.",
        "revisión humana de tickets de baja confianza y monitoreo semanal.": "human review of low-confidence tickets and weekly monitoring.",
        "Limitaciones conocidas": "Known limitations",
        "Los datos de entrenamiento son sintéticos, generados con reglas de negocio; el desempeño real se confirmará con tickets reales.": "Training data is synthetic, generated with business rules; real-world performance will be confirmed with actual tickets.",
        "La detección de críticos está cerca de la meta del 95%; por eso los casos dudosos pasan por revisión humana.": "Critical ticket detection is close to the 95% target; therefore, uncertain cases undergo human review.",
        "Sistemas o términos nuevos no se reconocen hasta reentrenar; el monitoreo los detecta y avisa.": "New systems or terms are not recognized until retraining; monitoring detects and alerts on them.",
        "Las recompensas de Q-Learning son supuestos de negocio: su política se usa como recomendación, no como regla.": "Q-Learning rewards are business assumptions: its policy is used as a recommendation, not a strict rule.",

        // Index Sections 4 to 8 || Secciones 4 a 8
        "Objetivos del proyecto": "Project Objectives",
        "Objetivo general": "General objective",
        "Objetivos específicos": "Specific objectives",
        "Antecedentes": "Background",
        "Marco teórico": "Theoretical framework",
        "Glosario": "Glossary",
        "Metodología y cronograma de desarrollo": "Development Methodology & Schedule",
        "Fase": "Phase",
        "Actividades": "Activities",
        "Ver": "View",
        "Abrir fase": "Open phase",
        "Cronograma (semanas del periodo académico)": "Schedule (academic term weeks)",
        "Productos esperados": "Expected Deliverables",
        "Referencias bibliográficas": "Bibliographic References",

        // CRISP-ML Common Headers & Tables || Encabezados y Tablas Comunes CRISP-ML
        "Fase 1 · Entendimiento del Negocio y los Datos": "Phase 1 · Business & Data Understanding",
        "Fase 2 · Ingeniería de Datos": "Phase 2 · Data Engineering",
        "Fase 3 · Ingeniería del Modelo": "Phase 3 · Model Engineering",
        "Fase 4 · Evaluación del Modelo": "Phase 4 · Model Evaluation",
        "Fase 5 · Despliegue del Modelo": "Phase 5 · Model Deployment",
        "Fase 6 · Monitoreo y Mantenimiento": "Phase 6 · Monitoring & Maintenance",
        "Objetivos de Negocio": "Business Objectives",
        "Necesidad": "Need",
        "Meta del proyecto ML": "ML Project Goal",
        "Criterio de éxito (KPI)": "Success Criterion (KPI)",
        "Preguntas Clave": "Key Questions",
        "Evaluación de Datos": "Data Evaluation",
        "registros": "records",
        "descripciones únicas": "unique descriptions",
        "duplicados": "duplicates",
        "palabras distintas": "distinct words",
        "palabras / ticket": "words / ticket",
        "test visto en train": "test seen in train",
        "Chequeos de calidad, relevancia y suficiencia": "Quality, relevance & sufficiency checks",
        "Chequeo": "Check",
        "Resultado": "Result",
        "Detalle": "Detail",
        "Cumple": "Pass",
        "No cumple": "Fail",
        "Distribución de la variable objetivo (prioridad) por categoría": "Target variable distribution (priority) by category",
        "Prioridad": "Priority",
        "Total": "Total",
        "Tickets reales registrados en la aplicación": "Real tickets registered in the application",
        "Análisis de Viabilidad": "Feasibility Analysis",
        "Aspecto": "Aspect",
        "Detalle / Mitigación": "Detail / Mitigation",
        "filas cargadas": "loaded rows",
        "filas eliminadas": "removed rows",
        "filas para modelar": "rows for modeling",
        "última ejecución": "last execution",
        "Selección de fuentes": "Source selection",
        "Fuente": "Source",
        "Archivo": "File",
        "Filas": "Rows",
        "Huella (sha256)": "Fingerprint (sha256)",
        "Limpieza": "Cleaning",
        "Paso": "Step",
        "Eliminadas": "Removed",
        "Filas resultantes": "Resulting rows",
        "Regla": "Rule",
        "Distribución después de limpiar": "Distribution after cleaning",
        "Transformación": "Transformation",
        "Ejemplos antes y después": "Examples before and after",
        "Texto original": "Original text",
        "Texto limpio": "Clean text",
        "Documentación y reproducibilidad": "Documentation & reproducibility",
        "Librería": "Library",
        "Versión al entrenar": "Version at training",
        "Versión actual": "Current version",
        "Coincide": "Matches",
        "modelo de prioridad": "priority model",
        "modelo de categoría": "category model",
        "validación cruzada": "cross-validation",
        "último entrenamiento": "last training",
        "Selección de algoritmos": "Algorithm selection",
        "Algoritmo": "Algorithm",
        "Por qué se incluye": "Why it is included",
        "Entrega probabilidad": "Outputs probability",
        "Sí": "Yes",
        "No": "No",
        "Entrenamiento": "Training",
        "Exactitud": "Accuracy",
        "Recall Crítico": "Critical Recall",
        "Tiempo": "Time",
        "Ajuste de hiperparámetros": "Hyperparameter tuning",
        "Combinaciones probadas": "Tested combinations",
        "Mejores parámetros (prioridad)": "Best parameters (priority)",
        "Sin ajuste": "Without tuning",
        "Con ajuste": "With tuning",
        "Umbral": "Threshold",
        "Redondeo normal": "Standard rounding",
        "Corte": "Cutoff",
        "Ver curva umbral vs. desempeño": "View threshold vs. performance curve",
        "Decisión y siguiente paso": "Decision & next step",
        "exactitud (test)": "accuracy (test)",
        "F1 macro": "Macro F1",
        "exactitud categoría": "category accuracy",
        "tickets de test": "test tickets",
        "Modelos evaluados: ¿cuál escogimos y por qué?": "Evaluated models: which one did we choose and why?",
        "Pregunta": "Question",
        "Modelo": "Model",
        "Tipo": "Type",
        "Cómo responde": "How it answers",
        "Elegido": "Selected",
        "Grupo": "Cluster",
        "Tickets": "Tickets",
        "Palabras que lo definen": "Defining keywords",
        "Prioridad más común": "Most common priority",
        "Política": "Policy",
        "Recompensa media": "Average reward",
        "Críticos dejados en cola": "Critical left in queue",
        "Situación": "Situation",
        "Acción aprendida": "Learned action",
        "Métricas de desempeño": "Performance metrics",
        "Criterio": "Criterion",
        "Mínimo": "Minimum",
        "Origen": "Origin",
        "Desempeño por prioridad": "Performance by priority",
        "Precisión": "Precision",
        "Validación cruzada vs. test": "Cross-validation vs. test",
        "Conjunto": "Set",
        "Análisis de riesgos": "Risk analysis",
        "Riesgo": "Risk",
        "Nivel": "Level",
        "Evidencia": "Evidence",
        "Mitigación": "Mitigation",
        "Sesgos: desempeño por subgrupo": "Bias: performance by subgroup",
        "Variable": "Variable",
        "Revisión con expertos": "Expert review",
        "errores revisados": "reviewed errors",
        "error del modelo": "model error",
        "error de la etiqueta": "label error",
        "ambiguos": "ambiguous",
        "exactitud ajustada": "adjusted accuracy",
        "Ticket": "Ticket",
        "Real": "Actual",
        "Confianza": "Confidence",
        "Veredicto": "Verdict",
        "Guardar": "Save",
        "No hay errores pendientes de revisión.": "No errors pending review.",
        "Implementación en producción": "Production deployment",
        "Verificación": "Check",
        "Detalle / cómo corregir": "Detail / how to fix",
        "Pendiente": "Pending",
        "Documentación y capacitación": "Documentation & training",
        "Protocolos: monitoreo, retroalimentación e incidencias": "Protocols: monitoring, feedback & incidents",
        "tickets clasificados": "classified tickets",
        "revisados por consultores": "reviewed by consultants",
        "acuerdo modelo-consultor": "model-consultant agreement",
        "sin clasificar": "unclassified",
        "Gestión de incidencias": "Incident management",
        "Cómo se detecta": "How it is detected",
        "Qué hacer": "What to do",
        "Supervisión continua": "Continuous monitoring",
        "Indicador": "Indicator",
        "Actual": "Current",
        "Línea base": "Baseline",
        "Nota": "Note",
        "Atención": "Warning",
        "Alerta": "Alert",
        "Evolución semanal": "Weekly evolution",
        "Semana": "Week",
        "Críticos": "Critical",
        "Revisados": "Reviewed",
        "Acuerdo": "Agreement",
        "Confianza media": "Average confidence",
        "Baja confianza": "Low confidence",
        "Detección de degradación (drift)": "Degradation detection (drift)",
        "Señal": "Signal",
        "Medida": "Metric",
        "Qué significa": "What it means",
        "Actualización del modelo": "Model update",
        "Criterio para reentrenar": "Retraining criterion",
        "Valor actual": "Current value",
        "¿Se cumple?": "Met?",
        "Historial de versiones": "Version history",
        "Fecha": "Date",
        "Versión": "Version",
        "Revisión de riesgos": "Risk review",
        "Riesgo / KPI": "Risk / KPI",
        "Evidencia actual": "Current evidence"
    };

    const placeholderDict = {
        "Usuario": "Username",
        "Contraseña": "Password",
        "Nueva clave": "New password",
        "Comentario (opcional)": "Comment (optional)",
        "Ej: Falla en proceso específico de facturación...": "Ex: Failure in specific billing process..."
    };

    // Parameterized & partial phrase replacements || Reemplazos parametrizados y parciales
    const phraseReplacements = [
        [/Ticket Triage ML: priorización y asignación automática de tickets de mesa de ayuda con aprendizaje automático/g, "Ticket Triage ML: automated helpdesk ticket prioritization and assignment with machine learning"],
        [/Sistema web que lee cada ticket, predice su prioridad con un modelo de Machine Learning y lo asigna al consultor adecuado, desarrollado con la metodología CRISP-ML\./g, "Web system that reads each ticket, predicts its priority using a Machine Learning model, and assigns it to the right consultant, built with the CRISP-ML methodology."],
        [/Una mesa de ayuda recibe a diario solicitudes de los usuarios de una organización con varias sedes y muchos sistemas: facturación, nómina, inventario, ERP, agendamiento de citas, historia clínica, VPN, entre otros\. Cada solicitud \(ticket\) tiene un título y una descripción en texto libre, y debe ser atendida por un consultor\./g, "A helpdesk receives daily requests from users across an organization with multiple branches and systems: billing, payroll, inventory, ERP, appointment scheduling, medical records, VPN, among others. Each request (ticket) has a title and free-text description, and must be handled by a consultant."],
        [/Usuario que reporta una falla o hace una solicitud\./g, "User who reports an issue or submits a request."],
        [/Atiende incidentes: algo dejó de funcionar\./g, "Handles incidents: something stopped working."],
        [/Atiende requerimientos: algo nuevo o un cambio\./g, "Handles requirements: something new or a change."],
        [/Gestiona usuarios y supervisa la operación y el modelo\./g, "Manages users and oversees operations and the model."],
        [/Los tickets se atienden por orden de llegada, sin una prioridad objetiva\./g, "Tickets are handled on a first-come, first-served basis without objective priority."],
        [/La prioridad depende de cómo escribe el usuario: casi todos dicen "urgente"\./g, "Priority depends on how the user writes: almost everyone says 'urgent'."],
        [/La asignación a consultores es manual, lo que genera retrasos y cargas desiguales\./g, "Assignment to consultants is manual, causing delays and uneven workloads."],
        [/No queda registro de qué tan bien se priorizó, así que el proceso no mejora con el tiempo\./g, "No record is kept of how well prioritization worked, so the process does not improve over time."],
        [/Medido con (\d+) tickets que el modelo nunca vio: exactitud ([\d.]+)%, F1 macro ([\d.]+)% y ([\d.]+)% de los tickets críticos detectados\./g, "Measured on $1 unseen test tickets: $2% accuracy, $3% macro F1, and $4% of critical tickets detected."],
        [/Apto con condiciones:/g, "Approved with conditions:"],
        [/Apto para despliegue/g, "Approved for deployment"],
        [/No apto/g, "Not approved"],
        [/\(puntaje 0-3\) \+ corte Crítico ([\d.]+)/g, "(score 0-3) + Critical cutoff $1"],
        [/\+ umbral Crítico ([\d.]+)/g, "+ Critical threshold $1"],
        [/\(grupo → prioridad mayoritaria\)/g, "(cluster → majority priority)"],
        [/· elegido/g, "· selected"],
        [/no supervisado/g, "unsupervised"],
        [/Decisión:/g, "Decision:"],
        [/Pregunta problema:/g, "Research question:"],
        [/Rol activo:/g, "Active role:"],
        [/Tickets asignados activos:/g, "Active assigned tickets:"],
        [/Consultor Tecnico/g, "Technical Consultant"],
        [/Consultor Funcional/g, "Functional Consultant"],
        [/Por qué se eligió, K-Means y Q-Learning: fase/g, "Why it was chosen, K-Means, and Q-Learning: phase"],
        [/Ticket Triage ML · Proyecto de Gestión del Conocimiento · Metodología CRISP-ML\(Q\) ·/g, "Ticket Triage ML · Knowledge Management Project · CRISP-ML(Q) Methodology ·"],
        [/Estado del modelo:/g, "Model status:"],
        [/Se recomienda reentrenar/g, "Retraining recommended"],
        [/Modelo elegido:/g, "Selected model:"],
        [/Por qué:/g, "Why:"],
        [/Entorno actual:/g, "Current environment:"],
        [/Modelo en uso:/g, "Model in use:"],
        [/Lista de verificación:/g, "Checklist:"],
        [/Manual técnico:/g, "Technical manual:"],
        [/Guía rápida por rol para las sesiones de formación:/g, "Quick role-based guide for training sessions:"]
    ];

    function translateKnown(text) {
        const normalized = text.trim().replace(/\s+/g, ' ');
        if (!normalized) return text;

        const leadingSpace = text.match(/^\s*/)[0];
        const trailingSpace = text.match(/\s*$/)[0];

        if (exactDict[normalized]) {
            return leadingSpace + exactDict[normalized] + trailingSpace;
        }

        let result = normalized;
        for (const [regex, replacement] of phraseReplacements) {
            result = result.replace(regex, replacement);
        }
        return leadingSpace + result + trailingSpace;
    }

    // Detect if remaining text still has Spanish content || Detectar si el texto restante aún tiene contenido en español
    function looksLikeSpanish(str) {
        const clean = str.trim();
        if (clean.length < 4) return false;
        if (/^[\d\s.,:%+\-–—/()[\]·⚠←→]+$/.test(clean)) return false;
        if (/[áéíóúüñ¿¡]/i.test(clean)) return true;
        return /\b(el|la|los|las|un|una|unos|unas|de|del|al|en|con|por|para|sin|sobre|entre|hasta|desde|según|mediante|que|como|cuando|donde|quien|cual|este|esta|estos|estas|ese|esa|esos|esas|su|sus|se|es|son|fue|ser|estar|está|están|tiene|tienen|puede|pueden|debe|deben|hace|cada|todo|toda|todos|todas|otro|otra|otros|otras|más|menos|muy|ya|no|sí)\b/i.test(clean);
    }

    function applyCachedToNodes(pendingNodes, src, translated) {
        for (const item of pendingNodes) {
            if (item.normalized === src && currentLang === 'en') {
                item.node.nodeValue = item.leading + translated + item.trailing;
            }
        }
    }

    // Fetch single string helper || Auxiliar para traducir cadena individual
    async function fetchSingle(src, pendingNodes) {
        try {
            const url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=es&tl=en&dt=t&q=' + encodeURIComponent(src);
            const resp = await fetch(url);
            if (!resp.ok) return;
            const data = await resp.json();
            const translated = (data && data[0]) ? data[0].map(part => part[0]).join('') : src;
            if (translated) {
                dynamicCache[src] = translated;
                applyCachedToNodes(pendingNodes, src, translated);
            }
        } catch (e) {}
    }

    // Parallel batched translation for instant first-time load || Traducción paralela en lote para carga inicial instantánea
    async function translateDynamicBatch(pendingNodes) {
        const uniqueTexts = [...new Set(pendingNodes.map(item => item.normalized))];
        const CHUNK_SIZE = 15;

        for (let i = 0; i < uniqueTexts.length; i += CHUNK_SIZE) {
            if (currentLang !== 'en') return;
            const chunk = uniqueTexts.slice(i, i + CHUNK_SIZE);
            const joined = chunk.join('\n@@\n');

            try {
                const url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=es&tl=en&dt=t&q=' + encodeURIComponent(joined);
                const resp = await fetch(url);
                if (!resp.ok) throw new Error('HTTP error');
                const data = await resp.json();
                const fullTranslated = (data && data[0]) ? data[0].map(part => part[0]).join('') : '';
                const parts = fullTranslated.split(/\s*@@\s*/);

                if (parts.length === chunk.length) {
                    chunk.forEach((src, idx) => {
                        const tr = parts[idx].trim();
                        if (tr) {
                            dynamicCache[src] = tr;
                            applyCachedToNodes(pendingNodes, src, tr);
                        }
                    });
                } else {
                    // Parallel fallback if delimiter count differs || Respaldo en paralelo si cambia el conteo del delimitador
                    await Promise.all(chunk.map(src => fetchSingle(src, pendingNodes)));
                }
            } catch (e) {
                await Promise.all(chunk.map(src => fetchSingle(src, pendingNodes)));
            }
        }

        try {
            localStorage.setItem(CACHE_KEY, JSON.stringify(dynamicCache));
        } catch (e) {}
    }

    function applyLanguage(lang) {
        currentLang = lang;
        document.documentElement.lang = lang;

        document.querySelectorAll('.lang-opt').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === lang);
        });

        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        const unmappedSpanish = [];
        let node;

        while ((node = walker.nextNode())) {
            const parent = node.parentElement;
            if (!parent || ['SCRIPT', 'STYLE', 'CODE', 'PRE'].includes(parent.tagName)) continue;
            if (parent.closest('.lang-switcher')) continue;

            if (!node._esOriginal) {
                node._esOriginal = node.nodeValue;
            }

            if (lang === 'es') {
                node.nodeValue = node._esOriginal;
                continue;
            }

            const step1Text = translateKnown(node._esOriginal);
            const norm = step1Text.trim().replace(/\s+/g, ' ');
            const lead = step1Text.match(/^\s*/)[0];
            const trail = step1Text.match(/\s*$/)[0];

            if (dynamicCache[norm]) {
                node.nodeValue = lead + dynamicCache[norm] + trail;
            } else {
                node.nodeValue = step1Text;
                if (looksLikeSpanish(norm)) {
                    unmappedSpanish.push({ node, normalized: norm, leading: lead, trailing: trail });
                }
            }
        }

        // Translate input placeholders || Traducir placeholders de inputs
        document.querySelectorAll('input[placeholder], textarea[placeholder]').forEach(el => {
            if (!el._esPlaceholder) {
                el._esPlaceholder = el.getAttribute('placeholder');
            }
            const orig = el._esPlaceholder;
            el.setAttribute('placeholder', lang === 'en' ? (placeholderDict[orig] || orig) : orig);
        });

        // Reveal page immediately after synchronous pass || Mostrar página inmediatamente tras la pasada síncrona
        document.documentElement.classList.remove('i18n-pending');

        // Translate any remaining uncached strings in parallel || Traducir cualquier cadena sin caché en paralelo
        if (lang === 'en' && unmappedSpanish.length > 0) {
            translateDynamicBatch(unmappedSpanish);
        }
    }

    window.setAppLanguage = function (lang) {
        applyLanguage(lang);
    };

    // Execute immediately if body is ready (eliminates DOMContentLoaded delay) || Ejecutar inmediatamente si body está listo
    if (document.body) {
        applyLanguage('en');
    } else {
        document.addEventListener('DOMContentLoaded', function () {
            applyLanguage('en');
        });
    }
})();