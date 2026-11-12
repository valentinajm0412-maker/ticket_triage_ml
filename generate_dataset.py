"""
Generador de dataset sintético de tickets || Synthetic ticket dataset generator

La prioridad NO sale de una plantilla fija: se calcula con reglas de negocio
(severidad + alcance + impacto para incidentes; tipo + urgencia para
requerimientos). Así el modelo tiene que aprender señales, no memorizar frases.
"""
import os
import random
import pandas as pd

SEED = 42
NOISE_RATE = 0.03  # % de etiquetas "mal puestas" a propósito (como en la vida real)

MODULOS = [
    'facturación electrónica', 'nómina', 'inventario', 'cartera', 'contabilidad',
    'portal de clientes', 'CRM', 'punto de venta', 'compras', 'tesorería',
    'agendamiento de citas', 'historia clínica', 'correo corporativo', 'VPN',
    'ERP', 'reportes gerenciales', 'pagos en línea', 'módulo de despachos',
]

# ---------------- INCIDENTES ----------------
# severidad: 3 = caída total, 2 = grave, 1 = parcial, 0 = cosmético
SINTOMAS = {
    3: ['el sistema de {m} está completamente caído', '{m} no responde',
        'no hay acceso a {m}', 'el servidor de {m} se cayó',
        '{m} muestra error 500 y no carga', 'perdimos conexión total con {m}'],
    2: ['{m} no deja guardar los registros', 'error al procesar transacciones en {m}',
        '{m} se bloquea al confirmar', 'no se pueden generar documentos en {m}',
        'los datos de {m} no se están sincronizando', '{m} rechaza todas las operaciones'],
    1: ['{m} está muy lento', 'error intermitente en {m}',
        'el cálculo de totales en {m} sale incorrecto', 'a veces {m} cierra la sesión',
        'la exportación a Excel desde {m} falla de vez en cuando',
        'las notificaciones de {m} llegan tarde',
        # Ambiguos: incidentes redactados como solicitud
        'solicito revisar {m} porque los totales no cuadran',
        'necesito que revisen {m}, no está enviando los correos'],
    0: ['un botón de {m} se ve desalineado', 'hay un error de ortografía en {m}',
        'el logo no carga en la pantalla de {m}', 'los colores de {m} se ven raros',
        'un texto en {m} aparece cortado'],
}
# alcance: 2 = toda la empresa, 1 = un área / varios, 0 = un usuario
ALCANCES = {
    2: ['afecta a toda la empresa', 'pasa en todas las sedes', 'ningún usuario puede trabajar',
        'le ocurre a todos los usuarios'],
    1: ['afecta al área de {a}', 'varios usuarios de {a} lo reportan',
        'pasa en la sede {s}', 'le ocurre a un grupo de usuarios'],
    0: ['solo me pasa a mí', 'le ocurre a un usuario', 'lo vi solo en mi equipo',
        'un compañero lo reportó'],
}
IMPACTOS = [  # suman +1 a la prioridad
    'estamos sin poder facturar', 'los clientes no pueden pagar',
    'hay pacientes esperando atención', 'se detuvieron los despachos',
    'el cierre de mes está bloqueado',
]

# ---------------- REQUERIMIENTOS ----------------
# base: 0 = consulta/estético, 1 = ajuste funcional
TIPOS_REQ = {
    0: ['quisiera saber cómo usar {m}', 'tengo una duda sobre {m}',
        'sugiero cambiar el color del menú de {m}', 'pido mejorar el texto de bienvenida de {m}',
        '¿dónde encuentro el manual de {m}?', 'propongo agregar un filtro en {m}'],
    1: ['solicito agregar un campo nuevo en {m}', 'necesito un reporte nuevo en {m}',
        'requiero permisos de supervisor en {m}', 'solicito crear un usuario en {m}',
        'necesitamos ajustar la plantilla de impresión de {m}',
        'solicito modificar el flujo de aprobación en {m}',
        # Ambiguos: requerimientos redactados como falla
        'no puedo ver la opción de aprobar en {m}, necesito acceso',
        'el reporte de {m} no trae la columna de centro de costo',
        'no me aparece el botón de exportar en {m}'],
}
URGENCIAS_REQ = [  # suman +1
    'es para el cierre de esta semana', 'lo pide auditoría con fecha límite',
    'es un requisito de la DIAN', 'lo necesitamos para un cliente nuevo que entra mañana',
]

AREAS = ['ventas', 'contabilidad', 'logística', 'recursos humanos', 'facturación', 'admisiones']
SEDES = ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'norte', 'centro']
SALUDOS = ['', '', 'Buenos días, ', 'Hola equipo, ', 'Buenas tardes, ', 'Por favor ayuda: ']
TIEMPOS = ['', '', ' desde esta mañana', ' desde hace una hora', ' desde ayer', ' hace unos minutos']
CIERRES = ['', '', ' Gracias.', ' Quedo atento.', ' Es urgente.', ' Adjunto pantallazo.']

PRIORIDADES = ['Bajo', 'Medio', 'Alto', 'Critico']

# Títulos genéricos: los usuarios reales no siempre dicen si es incidente o requerimiento
TITULOS_GENERICOS = ['Problema con {m}', '{m}', 'Ayuda con {m}', 'Caso {m}',
                     'Novedad en {m}', 'Revisión de {m}', 'Tema {m}', 'Urgente {m}']


def _titulo(modulo, especifico):
    if random.random() < 0.6:
        t = random.choice(TITULOS_GENERICOS).format(m=modulo)
    else:
        t = especifico
    return t[0].upper() + t[1:]


def _fill(text):
    return text.format(m=random.choice(MODULOS), a=random.choice(AREAS), s=random.choice(SEDES))


def _incidente():
    modulo = random.choice(MODULOS)
    sev = random.choices([3, 2, 1, 0], weights=[2, 3, 3, 2])[0]
    alc = random.choice([2, 1, 0])
    tiene_impacto = sev >= 1 and random.random() < 0.3

    score = sev + alc + (1 if tiene_impacto else 0)
    prioridad = 'Critico' if score >= 4 else 'Alto' if score == 3 else 'Medio' if score == 2 else 'Bajo'

    sintoma = random.choice(SINTOMAS[sev]).format(m=modulo)
    partes = [random.choice(SALUDOS) + sintoma + random.choice(TIEMPOS) + ',',
              _fill(random.choice(ALCANCES[alc]))]
    if tiene_impacto:
        partes.append('y ' + random.choice(IMPACTOS))
    desc = ' '.join(partes) + '.' + random.choice(CIERRES)
    titulo = _titulo(modulo, f'{"Caída" if sev == 3 else "Error" if sev >= 1 else "Detalle visual"} en {modulo}')
    return titulo, desc, 'Incidente', prioridad


def _requerimiento():
    modulo = random.choice(MODULOS)
    base = random.choice([0, 1])
    urgente = random.random() < 0.25

    score = base + (1 if urgente else 0)
    prioridad = ['Bajo', 'Medio', 'Alto'][score]

    tipo = random.choice(TIPOS_REQ[base]).format(m=modulo)
    desc = random.choice(SALUDOS) + tipo
    if urgente:
        desc += ', ' + random.choice(URGENCIAS_REQ)
    desc += '.' + random.choice(CIERRES)
    titulo = _titulo(modulo, f'{"Consulta" if base == 0 else "Solicitud"} sobre {modulo}')
    return titulo, desc, 'Requerimiento', prioridad


def _ruido(desc):
    """Simula errores de digitación: minúsculas, sin tildes, sin puntuación."""
    r = random.random()
    if r < 0.15:
        desc = desc.lower()
    elif r < 0.25:
        desc = desc.translate(str.maketrans('áéíóúÁÉÍÓÚ', 'aeiouAEIOU'))
    elif r < 0.30:
        desc = desc.replace(',', '').replace('.', '')
    return desc[0].upper() + desc[1:] if desc else desc


def generate_dataset(num_records=3000, seed=SEED):
    random.seed(seed)
    data, vistos = [], set()
    intentos = 0

    while len(data) < num_records and intentos < num_records * 20:
        intentos += 1
        gen = _incidente if random.random() < 0.6 else _requerimiento
        titulo, desc, categoria, prioridad = gen()
        desc = _ruido(desc)

        if desc in vistos:  # sin duplicados exactos
            continue
        vistos.add(desc)

        if random.random() < NOISE_RATE:  # ruido de etiqueta controlado
            prioridad = random.choice([p for p in PRIORIDADES if p != prioridad])
        if random.random() < NOISE_RATE:  # cliente que escoge mal la categoría
            categoria = 'Requerimiento' if categoria == 'Incidente' else 'Incidente'

        data.append({'titulo': titulo, 'descripcion': desc,
                     'categoria': categoria, 'prioridad': prioridad})

    os.makedirs('ml_models', exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv('ml_models/ticket_dataset.csv', index=False)
    print(f'Dataset creado con {len(df)} registros ({df["descripcion"].nunique()} únicos).')
    print(pd.crosstab(df['prioridad'], df['categoria'], margins=True))
    return df


if __name__ == '__main__':
    generate_dataset()
