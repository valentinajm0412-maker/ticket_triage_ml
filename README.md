# Ticket Triage ML

Mesa de ayuda que clasifica la **prioridad** de cada ticket con Machine Learning y lo asigna al consultor
con menor carga. El proyecto sigue la metodología **CRISP-ML** y cada fase tiene su propia hoja en la app
(menú **CRISP-ML**, solo Admin).

## Requisitos

- Python 3.12 o superior (probado con 3.14 en Windows).
- Dependencias con versiones fijas en `requirements.txt`. Los modelos `.pkl` deben cargarse con las
  mismas versiones de pandas y scikit-learn con que se entrenaron.

## Estructura del sitio

- **Sitio público (sin login):** `/` presenta el proyecto con los campos del formato de propuesta PGC
  (datos generales, entendimiento del problema, problema y justificación, solución, objetivos, marco referencial,
  metodología y cronograma, productos y referencias) y `/crisp/fase1` a `/crisp/fase6` muestran cada fase CRISP-ML.
  El contenido se edita en `project_info.py`.
- **Aplicación (con login):** `/login`, `/dashboard`, `/create_ticket` y `/admin/users`. Las acciones que modifican
  datos en las hojas CRISP-ML (revisión de expertos, exportar retroalimentación) solo las ve el Admin.

## Modelo elegido

La prioridad se predice con **Linear Regression ordinal**: la prioridad se trata como un puntaje
(Bajo=0, Medio=1, Alto=2, Crítico=3) y por encima de un corte ajustado en validación cruzada el ticket es Crítico.
Se eligió frente a Random Forest, Logistic Regression, KNN y K-Means por tener el mayor F1 macro en
validación cruzada cumpliendo la meta de detección de críticos. K-Means se usa para entender qué tipos de
tickets llegan y Q-Learning para recomendar la acción. El detalle está en la hoja **Evaluación del Modelo**.

## Instalación (Windows, PowerShell)

```
git clone https://github.com/valentinajm0412-maker/ticket_triage_ml.git
cd ticket_triage_ml
py -m pip install -r requirements.txt
py generate_dataset.py      # 3000 tickets sintéticos
py train_models.py          # Fases 2, 3 y 4: limpia, entrena, evalúa y compara modelos (2-4 min)
py create_admin.py          # usuario admin / admin123 (cambiarla luego)
py seed_users.py            # consultores y cliente de prueba (clave 123456)
py app.py                   # desarrollo: http://127.0.0.1:5000 (sitio público) y /login (app)
```

Para que la app se recargue sola al editar código en desarrollo: `$env:FLASK_DEBUG = "1"` antes de `py app.py`.

## Producción

```
$env:SECRET_KEY = "una-clave-larga-y-aleatoria"
$env:APP_ENV = "production"
py run_production.py        # servidor waitress en http://0.0.0.0:8000
```

| Variable | Uso | Por defecto |
|---|---|---|
| `SECRET_KEY` | Firma de sesiones. **Obligatoria en producción** | aleatoria (se pierde al reiniciar) |
| `APP_ENV` | `production` desactiva debug y escucha en 0.0.0.0 | `development` |
| `DATABASE_URL` | Base de datos | `sqlite:///ticket_triage.db` |
| `PORT` | Puerto | 5000 (dev) / 8000 (prod) |
| `LOW_CONFIDENCE` | Bajo esta probabilidad el ticket se marca ⚠ para revisión | 0.5 |

Monitoreo: `GET /health` devuelve el estado de la base de datos y del modelo (200 ok / 503 degradado).
Cada predicción se registra en `logs/predictions.log`.

## Ciclo de mejora continua (Fase 6: Monitoreo y Mantenimiento)

1. El modelo asigna prioridad y marca con ⚠ los tickets de baja confianza.
2. El consultor confirma o corrige la prioridad desde el Dashboard.
3. La hoja **Monitoreo y Mantenimiento** compara los últimos 30 días contra la línea base de la Evaluación:
   acuerdo modelo-consultor, confianza, drift (PSI y palabras nuevas) y KPI de negocio.
4. Cuando se cumple un criterio de reentrenamiento: `py retrain.py`
   - exporta la retroalimentación a `ml_models/real_tickets.csv`,
   - respalda el modelo actual en `ml_models/backup/`,
   - reentrena y compara; si el nuevo empeora (recall de Crítico o F1 bajan más de 2 puntos), restaura el anterior.
5. Reiniciar la app para cargar el modelo nuevo. Cada versión queda en `ml_models/model_history.json`.

Para probar el monitoreo sin datos reales: `py simulate_production.py` (o `--drift` para simular
sistemas nuevos; `--reset` borra lo simulado). No usar en producción.

Rutina: **semanal** revisar Monitoreo y los tickets ⚠ · **mensual** `py retrain.py` · **trimestral** revisar KPI y umbrales.

## Estructura

| Archivo | Fase CRISP-ML | Qué hace |
|---|---|---|
| `crisp_fase1.py` | 1. Entendimiento del negocio y los datos | Objetivos, KPI, evaluación del dataset |
| `generate_dataset.py` | 2. Ingeniería de datos | Dataset sintético con reglas de negocio |
| `data_pipeline.py` | 2. Ingeniería de datos | Selección, limpieza, transformación y reporte |
| `model_engineering.py` | 3. Ingeniería del modelo | Validación cruzada, GridSearch, umbral de críticos |
| `train_models.py` | 2-3-4 | Orquesta el entrenamiento completo |
| `evaluation.py`, `evaluate_model.py` | 4. Evaluación | Métricas en test, riesgos, sesgos, errores para expertos |
| `model_comparison.py`, `compare_models.py` | 4. Evaluación | Compara Logistic Regression, KNN, K-Means, Linear Regression y Q-Learning |
| `config.py`, `deployment.py`, `run_production.py`, `export_feedback.py` | 5. Despliegue | Configuración, verificación, retroalimentación |
| `monitoring.py`, `retrain.py`, `simulate_production.py` | 6. Monitoreo y mantenimiento | Supervisión, drift, reentrenamiento seguro, simulación |
| `app.py`, `triage_service.py`, `models.py` | App | Rutas, predicción + asignación, base de datos |
| `project_info.py` | Sitio público | Contenido de la presentación del proyecto (formato de propuesta PGC) |

Reportes generados en `ml_models/`: `data_report.json`, `model_report.json`, `evaluation_report.json`, `model_history.json`.

## Roles

- **Cliente**: crea tickets y ve los suyos.
- **Consultor Técnico / Funcional**: atiende incidentes / requerimientos, confirma o corrige prioridades.
- **Admin**: ve todo, gestiona usuarios y las hojas CRISP-ML.
