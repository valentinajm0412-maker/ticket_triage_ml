import pandas as pd
import random

# Synthetic data generation for ML training || Generación de datos sintéticos para entrenamiento de ML
def generate_dataset(num_records=800):
    data = []
    
    # Vocabulary by priority level based on business rules || Vocabulario por nivel de prioridad según reglas de negocio
    critical_phrases = ['sistema completamente caído', 'no hay acceso a producción', 'operatividad de todo el sistema detenida', 'falla crítica en servidor principal']
    high_phrases = ['falla en proceso específico de facturación', 'módulo contable no funciona', 'error al procesar pagos', 'proceso específico inoperativo']
    medium_phrases = ['error de procesamiento de datos', 'cálculo incorrecto en reporte', 'lentitud al exportar archivo', 'falla en procesamiento sin afectar operatividad']
    low_phrases = ['duda sobre funcionalidad', 'solicitud para cambiar color del menú', 'actualizar texto de bienvenida', 'requerimiento general de interfaz']

    for _ in range(num_records):
        rand_val = random.random()
        
        # Rule assignment || Asignación de reglas
        if rand_val < 0.15:
            desc = random.choice(critical_phrases)
            priority = 'Critico'
            category = 'Incidente'
        elif rand_val < 0.35:
            desc = random.choice(high_phrases)
            priority = 'Alto'
            category = 'Incidente'
        elif rand_val < 0.70:
            desc = random.choice(medium_phrases)
            priority = 'Medio'
            category = random.choice(['Incidente', 'Requerimiento'])
        else:
            desc = random.choice(low_phrases)
            priority = 'Bajo'
            category = 'Requerimiento'
            
        data.append({'descripcion': desc, 'categoria': category, 'prioridad': priority})
        
    df = pd.DataFrame(data)
    
    # Save to the ML models folder || Guardar en la carpeta de modelos de ML
    df.to_csv('ml_models/ticket_dataset.csv', index=False)
    print("Dataset created successfully in ml_models/ticket_dataset.csv || Dataset creado con éxito en ml_models/ticket_dataset.csv")

if __name__ == '__main__':
    generate_dataset()