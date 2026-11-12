import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. Cargar el dataset
df = pd.read_csv('ml_models/ticket_dataset.csv')
X = df['descripcion']
y_priority = df['prioridad']
y_category = df['categoria']

# 2. Dividir ANTES de vectorizar (evita Data Leakage) con estratificación
X_train_raw, X_test_raw, y_train_p, y_test_p, y_train_c, y_test_c = (
    train_test_split(
        X, y_priority, y_category, test_size=0.2, random_state=42, stratify=y_priority
    )
)

# 3. Vectorizar ajustando SOLO con los datos de entrenamiento
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), strip_accents='unicode', lowercase=True
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


def train_best_classifier(X_tr, y_tr, X_te, y_te, target_name):
  models = {
      'Naive Bayes': MultinomialNB(),
      'Random Forest': RandomForestClassifier(random_state=42),
      'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
  }

  best_model = None
  best_accuracy = 0
  best_name = ''

  print(f'=== Entrenando modelos para: {target_name.upper()} ===\n')

  for name, model in models.items():
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    accuracy = accuracy_score(y_te, predictions)
    print(f'{name} Accuracy: {accuracy:.4f}')

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      best_model = model
      best_name = name

  print(
      f'\nMejor modelo para {target_name}: {best_name} ({best_accuracy:.4f})'
  )
  print(classification_report(y_te, best_model.predict(X_te)))
  return best_model


# 4. Entrenar modelo de Prioridad y de Categoría
best_priority_model = train_best_classifier(
    X_train, y_train_p, X_test, y_test_p, 'Prioridad'
)
best_category_model = train_best_classifier(
    X_train, y_train_c, X_test, y_test_c, 'Categoría'
)

# 5. Guardar los modelos y el vectorizador
joblib.dump(best_priority_model, 'ml_models/best_ticket_model.pkl')
joblib.dump(best_category_model, 'ml_models/best_category_model.pkl')
joblib.dump(vectorizer, 'ml_models/tfidf_vectorizer.pkl')
print('Modelos y vectorizador guardados con éxito en ml_models/')
