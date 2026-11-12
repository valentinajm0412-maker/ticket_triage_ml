"""
Configuración por variables de entorno (Fase 5: Despliegue) || Environment-based configuration

Desarrollo: no hace falta configurar nada.
Producción (PowerShell, antes de arrancar):
    $env:SECRET_KEY = "una-clave-larga-y-aleatoria"
    $env:APP_ENV = "production"
"""
import os
import secrets

APP_ENV = os.environ.get('APP_ENV', 'development')
IS_PRODUCTION = APP_ENV == 'production'

SECRET_KEY = os.environ.get('SECRET_KEY')
SECRET_KEY_FROM_ENV = bool(SECRET_KEY)
if not SECRET_KEY:
    # Clave temporal: las sesiones se cierran al reiniciar. En producción es obligatorio definirla.
    SECRET_KEY = secrets.token_hex(32)

DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1' and not IS_PRODUCTION
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///ticket_triage.db')
HOST = os.environ.get('HOST', '127.0.0.1' if not IS_PRODUCTION else '0.0.0.0')
PORT = int(os.environ.get('PORT', '5000' if not IS_PRODUCTION else '8000'))

# Tickets con probabilidad menor a este valor se marcan para revisión humana (viene de la Fase 4)
LOW_CONFIDENCE = float(os.environ.get('LOW_CONFIDENCE', '0.5'))

LOG_DIR = os.environ.get('LOG_DIR', 'logs')
VALID_STATUSES = ['Abierto', 'En Proceso', 'Resuelto']
VALID_PRIORITIES = ['Critico', 'Alto', 'Medio', 'Bajo']
