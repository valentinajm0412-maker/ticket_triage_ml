import os
import random
import pandas as pd


def generate_dataset(num_records=1200):
  data = []

  # Componentes combinables por nivel
  templates = {
      'Critico': {
          'categoria': ['Incidente'],
          'inicio': [
              'Sistema completamente caído',
              'Sin acceso a producción',
              'Operatividad detenida',
              'Error fatal',
              'Servidor principal sin respuesta',
          ],
          'detalle': [
              'para toda la empresa',
              'en todas las sedes',
              'desde hace 10 minutos',
              'bloqueando ventas y operaciones',
              'arrojando error 500 general',
          ],
      },
      'Alto': {
          'categoria': ['Incidente'],
          'inicio': [
              'Falla grave en',
              'No funciona el módulo de',
              'Error al procesar en',
              'Bloqueo total en',
              'Caída parcial de',
          ],
          'detalle': [
              'facturación electrónica',
              'pagos con tarjeta',
              'cierre contable',
              'autenticación de usuarios',
              'sincronización de inventario',
          ],
      },
      'Medio_Incidente': {
          'prioridad': 'Medio',
          'categoria': ['Incidente'],
          'inicio': [
              'Lentitud al cargar',
              'Cálculo incorrecto en',
              'Error intermitente al exportar',
              'Falla visual en',
          ],
          'detalle': [
              'el reporte mensual',
              'la vista de clientes',
              'el archivo Excel',
              'las notificaciones por correo',
          ],
      },
      'Medio_Requerimiento': {
          'prioridad': 'Medio',
          'categoria': ['Requerimiento'],
          'inicio': [
              'Solicitud de ajuste en',
              'Requerimiento para modificar',
              'Actualización de permisos en',
              'Configuración de nuevo campo en',
          ],
          'detalle': [
              'el formulario de registro',
              'el reporte de ventas',
              'el perfil de supervisor',
              'la plantilla de impresión',
          ],
      },
      'Bajo': {
          'categoria': ['Requerimiento'],
          'inicio': [
              'Duda sobre cómo usar',
              'Consulta general de',
              'Solicitud estética para cambiar',
              'Sugerencia de mejora en',
          ],
          'detalle': [
              'el color del menú lateral',
              'el texto de bienvenida',
              'los filtros de búsqueda',
              'el manual de usuario',
          ],
      },
  }

  for _ in range(num_records):
    rand_val = random.random()

    if rand_val < 0.20:
      key = 'Critico'
    elif rand_val < 0.45:
      key = 'Alto'
    elif rand_val < 0.62:
      key = 'Medio_Incidente'
    elif rand_val < 0.78:
      key = 'Medio_Requerimiento'
    else:
      key = 'Bajo'

    grupo = templates[key]
    desc = f"{random.choice(grupo['inicio'])} {random.choice(grupo['detalle'])}"
    priority = grupo.get('prioridad', key)
    category = random.choice(grupo['categoria'])

    data.append(
        {'descripcion': desc, 'categoria': category, 'prioridad': priority}
    )

  os.makedirs('ml_models', exist_ok=True)
  df = pd.DataFrame(data)
  df.to_csv('ml_models/ticket_dataset.csv', index=False)
  print(f'Dataset creado con {len(df)} registros.')
  print(df['prioridad'].value_counts())


if __name__ == '__main__':
  generate_dataset()
