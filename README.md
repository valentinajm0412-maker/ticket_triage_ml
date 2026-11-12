# Ticket Triage ML

Mesa de ayuda que clasifica la **prioridad** de cada ticket con Machine Learning y lo asigna al consultor
con menor carga. El proyecto sigue la metodología **CRISP-ML** y cada fase tiene su propia hoja en la app
(menú **CRISP-ML**, solo Admin).

## Requisitos

- Python 3.12 o superior (probado con 3.14 en Windows).
- Dependencias con versiones fijas en `requirements.txt`. Los modelos `.pkl` deben cargarse con las
  mismas versiones de pandas y scikit-learn con que se entrenaron.

## Instalación (Windows, PowerShell)

```
git clone https://github.com/valentinajm0412-maker/ticket_triage_ml.git
cd ticket_triage_ml
py -m pip install -r requirements.txt
py generate_dataset.py      # 3000 tickets sintéticos
py train_models.py          # Fases 2, 3 y 4: limpia, entrena, evalúa (1-3 min)
py create_admin.py          # usuario admin / admin123 (cambiarla luego)
py seed_users.py            # consultores y cliente de prueba (clave 123456)
py app.py                   # desarrollo: http://127.0.0.1:5000
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

## Ciclo de mejora continua

1. El modelo asigna prioridad y marca con ⚠ los tickets de baja confianza.
2. El consultor confirma o corrige la prioridad desde el Dashboard.
3. El Admin exporta la retroalimentación (hoja Despliegue o `py export_feedback.py`) → `ml_models/real_tickets.csv`.
4. `py train_models.py` reentrena usando datos sintéticos + reales y regenera los reportes.
5. Revisar la hoja Evaluación antes de subir el modelo nuevo.

## Estructura

| Archivo | Fase CRISP-ML | Qué hace |
|---|---|---|
| `crisp_fase1.py` | 1. Entendimiento del negocio y los datos | Objetivos, KPI, evaluación del dataset |
| `generate_dataset.py` | 2. Ingeniería de datos | Dataset sintético con reglas de negocio |
| `data_pipeline.py` | 2. Ingeniería de datos | Selección, limpieza, transformación y reporte |
| `model_engineering.py` | 3. Ingeniería del modelo | Validación cruzada, GridSearch, umbral de críticos |
| `train_models.py` | 2-3-4 | Orquesta el entrenamiento completo |
| `evaluation.py`, `evaluate_model.py` | 4. Evaluación | Métricas en test, riesgos, sesgos, errores para expertos |
| `config.py`, `deployment.py`, `run_production.py`, `export_feedback.py` | 5. Despliegue | Configuración, verificación, retroalimentación |
| `app.py`, `triage_service.py`, `models.py` | App | Rutas, predicción + asignación, base de datos |

Reportes generados en `ml_models/`: `data_report.json`, `model_report.json`, `evaluation_report.json`.

## Roles

- **Cliente**: crea tickets y ve los suyos.
- **Consultor Técnico / Funcional**: atiende incidentes / requerimientos, confirma o corrige prioridades.
- **Admin**: ve todo, gestiona usuarios y las hojas CRISP-ML.
