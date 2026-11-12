# Test Dataset Seeder (100 Diverse Tickets) || Semilla de Datos de Prueba (100 Tickets Variados)
import random
from datetime import datetime, timedelta
from app import app, db
from models import User, Ticket

try:
    import config
    VALID_PRIORITIES = list(getattr(config, "VALID_PRIORITIES", ["Critico", "Alto", "Medio", "Bajo"]))
except Exception:
    VALID_PRIORITIES = ["Critico", "Alto", "Medio", "Bajo"]


def resolve_priority_label(level: str) -> str:
    """Match configured priority spelling (e.g., Critico vs Crítico) || Ajustar ortografía según config.py"""
    for p in VALID_PRIORITIES:
        if p.lower().replace("í", "i") == level.lower().replace("í", "i"):
            return p
    return VALID_PRIORITIES[0]


# Base templates for realistic Incidents and Requirements || Plantillas base para Incidentes y Requerimientos realistas
INCIDENT_TEMPLATES = [
    ("Critico", "Caída total del servidor de facturación electrónica", "El servicio de facturación DIAN no responde en ninguna sede y detuvo la operación comercial."),
    ("Critico", "Base de datos principal del ERP inaccesible", "Error de conexión masivo al intentar abrir el módulo financiero y contable en toda la empresa."),
    ("Critico", "Falla crítica en sistema de historia clínica", "Los médicos no pueden guardar ni consultar evoluciones de pacientes en urgencias."),
    ("Critico", "Interrupción completa de VPN corporativa", "Ningún usuario remoto logra autenticarse en la red privada desde primera hora."),
    ("Critico", "Bloqueo total en pasarela de pagos y recaudo", "Las transacciones con tarjeta y transferencia arrojan error 500 en todas las cajas."),
    ("Alto", "Error grave al liquidar nómina quincenal", "El cálculo de deducciones y seguridad social genera valores inconsistentes para el área operativa."),
    ("Alto", "Falla en sincronización de inventario entre bodegas", "Las salidas de mercancía no descuentan stock en tiempo real y generan descuadres."),
    ("Alto", "Servidor de correo corporativo con demoras severas", "Los correos externos tardan más de dos horas en salir o rebotan sin entrega."),
    ("Alto", "Error al generar certificados de retención en la fuente", "El sistema arroja excepción al exportar el PDF masivo para proveedores."),
    ("Alto", "Caída intermitente en módulo de agendamiento de citas", "Se pierden reservas confirmadas cuando hay más de veinte usuarios simultáneos."),
    ("Alto", "Fallo de autenticación SSO en portal interno", "Varios departamentos reportan cierre inesperado de sesión cada cinco minutos."),
    ("Medio", "Lentitud al consultar reportes históricos de ventas", "Las consultas trimestrales tardan más de 40 segundos en cargar en pantalla."),
    ("Medio", "Error parcial al exportar informes a Excel", "Las columnas de fechas se descargan sin formato en el reporte de cartera."),
    ("Medio", "Impresora térmica de etiquetas desconfigurada en despacho", "Los códigos de barras salen cortados al imprimir guías desde el módulo logístico."),
    ("Medio", "Falla en notificación automática de vencimiento de contratos", "Las alertas por correo no se enviaron esta semana al área jurídica."),
    ("Medio", "Error al adjuntar soportes PDF mayores a 5 MB", "El formulario de radicación muestra mensaje de tiempo agotado al subir anexos."),
    ("Medio", "Descuadre menor en visualización de saldos de caja menor", "El resumen diario muestra diferencia de centavos por redondeo decimal."),
    ("Medio", "Fallo en filtro de búsqueda por cédula en directorio", "Al buscar usuarios con puntos o espacios no devuelve resultados."),
    ("Bajo", "Detalle visual en alineación de botones del menú lateral", "En pantallas pequeñas el texto del submenú se superpone ligeramente con el borde."),
    ("Bajo", "Error ortográfico en mensaje de confirmación de guardado", "Aparece una palabra sin tilde en la alerta verde tras actualizar perfil."),
    ("Bajo", "Ícono de ayuda no carga tooltip en vista de consulta", "Al pasar el cursor sobre el signo de interrogación no se despliega la guía rápida."),
    ("Bajo", "Color de etiqueta desactualizado en historial de auditoría", "El estado archivado se muestra en azul claro en lugar de gris neutro."),
    ("Bajo", "Ordenamiento por defecto en tabla de proveedores", "La lista aparece ordenada por ID interno en vez de orden alfabético."),
]

REQUIREMENT_TEMPLATES = [
    ("Critico", "Habilitación urgente de resolución DIAN para facturación", "Se requiere parametrizar hoy la nueva numeración autorizada antes de que venza la actual."),
    ("Critico", "Ajuste obligatorio por reforma tributaria en retenciones", "Actualización normativa de tarifas de IVA y retención requerida para cierre fiscal inmediato."),
    ("Critico", "Apertura de permisos de emergencia para auditoría externa", "Crear perfiles de lectura controlada para auditores regulatorios con ingreso hoy."),
    ("Alto", "Nuevo reporte consolidado para cierre contable mensual", "Desarrollar informe con cruce de cuentas por pagar y centros de costo para gerencia."),
    ("Alto", "Integración automática de órdenes de compra con ERP", "Automatizar la carga de pedidos aprobados sin digitación manual del área de compras."),
    ("Alto", "Creación masiva de usuarios para nueva sede regional", "Configurar 35 cuentas con roles operativos y acceso a módulos de atención."),
    ("Alto", "Flujo de aprobación en dos niveles para anticipos", "Agregar validación de jefe de área y dirección financiera antes de desembolso."),
    ("Alto", "Parametrización de nuevos conceptos de horas extras en nómina", "Incluir recargos nocturnos y festivos según actualización del reglamento interno."),
    ("Alto", "Exportación de archivo plano para dispersión bancaria", "Generar estructura compatible con el portal empresarial del banco."),
    ("Medio", "Agregar filtro por rango de fechas en tablero de gestión", "Permitir filtrar métricas semanales y mensuales desde la vista principal."),
    ("Medio", "Solicitud de cambio de rol para analista de cartera", "Actualizar permisos de consulta a edición en el módulo de recaudo."),
    ("Medio", "Incluir columna de centro de costo en reporte de gastos", "Añadir el código del departamento en la vista previa y descarga CSV."),
    ("Medio", "Configuración de plantilla de correo para bienvenida de clientes", "Actualizar logotipo y datos de contacto en las notificaciones automáticas."),
    ("Medio", "Habilitar validación de campo obligatorio en teléfono móvil", "Evitar el registro de solicitudes sin número de contacto válido."),
    ("Medio", "Restablecimiento de credenciales y actualización de perfil", "Cambio de contraseña y actualización de cargo para coordinador operativo."),
    ("Medio", "Consulta sobre parametrización de impuestos municipales", "Revisión funcional de tablas de ICA aplicables a sucursales nuevas."),
    ("Bajo", "Cambio de nombre en etiqueta de campo de observaciones", "Renombrar el campo 'Notas' a 'Observaciones adicionales' en el formulario."),
    ("Bajo", "Solicitud de manual de usuario para módulo de reportes", "Compartir guía paso a paso para la generación de extractos mensuales."),
    ("Bajo", "Ajuste de paginación a 25 registros por vista en listados", "Ampliar el número de filas visibles por página para facilitar la lectura."),
    ("Bajo", "Personalización de pie de página en formatos impresos", "Incluir dirección actualizada de la oficina principal en comprobantes."),
    ("Bajo", "Consulta general sobre horarios de mantenimiento programado", "Información sobre ventanas de actualización del sistema para el próximo mes."),
    ("Bajo", "Agregar opción de exportar gráfica resumen en formato PNG", "Permitir descargar la imagen estadística desde el panel de indicadores."),
]

BRANCHES = ["Sede Norte", "Sede Centro", "Sede Sur", "Sede Occidente", "Operación Nacional"]


def seed_100_tickets():
    with app.app_context():
        db.create_all()

        # Ensure users exist || Asegurar que existan usuarios
        users = User.query.all()
        if not users:
            print("No users found. Run 'python create_admin.py' and 'python seed_users.py' first.")
            return

        clients = [u for u in users if u.role == "Cliente"] or users
        tech_consultants = [u for u in users if u.role == "Consultor Tecnico"] or users
        func_consultants = [u for u in users if u.role == "Consultor Funcional"] or users
        admins = [u for u in users if u.role == "Admin"] or users

        existing_count = Ticket.query.count()
        if existing_count >= 100:
            print(f"Database already has {existing_count} tickets. No duplicates added.")
            return

        random.seed(42)
        tickets_to_create = 100 - existing_count
        columns = set(Ticket.__table__.columns.keys())

        # Track dynamic workload for balanced assignment || Controlar carga dinámica para asignación equilibrada
        workload_map = {u.id: (u.workload or 0) for u in users}
        status_pool = ["Abierto"] * 45 + ["En Proceso"] * 25 + ["Resuelto"] * 30
        random.shuffle(status_pool)

        created = 0
        for i in range(tickets_to_create):
            is_incident = (i % 2 == 0)
            category = "Incidente" if is_incident else "Requerimiento"
            pool = INCIDENT_TEMPLATES if is_incident else REQUIREMENT_TEMPLATES
            raw_prio, base_title, base_desc = pool[i % len(pool)]

            priority = resolve_priority_label(raw_prio)
            branch = BRANCHES[i % len(BRANCHES)]
            case_num = existing_count + i + 1
            title = f"{base_title} ({branch} #{case_num})"
            description = f"{base_desc} Reportado desde {branch} - Caso de prueba #{case_num}."

            status = status_pool[i % len(status_pool)]
            client = clients[i % len(clients)]

            # Select consultant with lowest current workload || Elegir consultor con menor carga actual
            candidates = tech_consultants if is_incident else func_consultants
            assignee = min(candidates, key=lambda u: workload_map[u.id])

            if status != "Resuelto":
                workload_map[assignee.id] += 1

            confidence = round(random.uniform(0.61, 0.98), 2)
            needs_review = confidence < 0.72 and status != "Resuelto"
            reviewed_by = None
            reviewed_at = None

            if status == "Resuelto" or (i % 6 == 0 and status == "En Proceso"):
                reviewed_by = assignee.id if assignee else admins[0].id
                reviewed_at = datetime.utcnow() - timedelta(hours=random.randint(1, 72))
                needs_review = False

            ticket = Ticket()
            field_values = {
                "title": title,
                "description": description,
                "category": category,
                "priority": priority,
                "priority_predicted": priority,
                "priority_confidence": confidence,
                "needs_review": needs_review,
                "priority_reviewed_by": reviewed_by,
                "priority_reviewed_at": reviewed_at,
                "status": status,
                "client_id": client.id,
                "assigned_to": assignee.id,
                "created_at": datetime.utcnow() - timedelta(days=random.randint(0, 20), hours=random.randint(0, 23)),
            }

            for col_name, val in field_values.items():
                if col_name in columns:
                    setattr(ticket, col_name, val)

            db.session.add(ticket)
            created += 1

        # Update consultant workloads || Actualizar cargas de trabajo de los consultores
        for u in users:
            if hasattr(u, "workload"):
                u.workload = workload_map[u.id]

        db.session.commit()
        print(f"Successfully created {created} test tickets. Total in DB: {Ticket.query.count()}")


if __name__ == "__main__":
    seed_100_tickets()