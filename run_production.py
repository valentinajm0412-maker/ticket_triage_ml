"""
Servidor de producción (Fase 5: Despliegue) || Production server

Windows y Linux:
    $env:SECRET_KEY = "una-clave-larga-y-aleatoria"
    $env:APP_ENV = "production"
    py run_production.py

En Linux también se puede usar: gunicorn -w 2 -b 0.0.0.0:8000 app:app
"""
from waitress import serve

import config
from app import app

if __name__ == '__main__':
    if not config.SECRET_KEY_FROM_ENV:
        print('ADVERTENCIA: SECRET_KEY no está definida; las sesiones se cerrarán al reiniciar.')
    print(f'Ticket Triage ML en http://{config.HOST}:{config.PORT} (health: /health)')
    serve(app, host=config.HOST, port=config.PORT, threads=8)
