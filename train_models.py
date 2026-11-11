import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the synthetic dataset || Cargar el dataset sintético
df = pd.read_csv('ml_models/ticket_dataset.csv')

# Separate features (text) and target labels (priority) || Separar características (texto) y etiquetas objetivo (prioridad)
X = df['descripcion']
y = df['prioridad']

# Convert text to numerical data using TF-IDF || Convertir texto a datos numéricos usando TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test) || Dividir datos en conjuntos de entrenamiento y prueba (80% entrena, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize the 3 Machine Learning models || Inicializar los 3 modelos de Machine Learning
models = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("Training models... || Entrenando modelos...\n")

# Train and evaluate each model || Entrenar y evaluar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name} Accuracy || Precisión: {accuracy:.4f}")
    
    # Select the best performing model || Seleccionar el modelo con mejor rendimiento
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest model selected: {best_model_name} with {best_accuracy:.4f} accuracy || Mejor modelo seleccionado: {best_model_name} con {best_accuracy:.4f} de precisión")

# Save the best model and the vectorizer for later use in the app || Guardar el mejor modelo y el vectorizador para uso posterior en la app
joblib.dump(best_model, 'ml_models/best_ticket_model.pkl')
joblib.dump(vectorizer, 'ml_models/tfidf_vectorizer.pkl')
print("\nModel and vectorizer saved successfully in ml_models/ || Modelo y vectorizador guardados con éxito en ml_models/")