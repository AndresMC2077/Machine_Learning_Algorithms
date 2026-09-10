#implementación de random forest con un framework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# Importes añadidos para el pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#construir el dataframe
df = pd.read_csv("dataset_compra_coche.csv")

# Nota: Debe ser una lista de listas para el encoder
orden_precio = [['Alto', 'Medio', 'Bajo']]
orden_kilometraje = [['Alto', 'Medio', 'Bajo']]
orden_antiguedad = [['Nuevo', 'Seminuevo', 'Usado']]
orden_marca = [['Premium', 'Media', 'Economica']]
orden_binarias = [['No', 'Si'], ['No', 'Si']] # Añadido para procesar las variables de "Si/No"

# AJustar y transformar otras columnas (Solo procesamos y (target) manualmente)
df['Comprar_num'] = df['Comprar'].map({'Si': 1, 'No': 0})

X = df.drop(columns=["Comprar_num", "Comprar"])  # Características (features) sin dropear las categóricas originales
y = df["Comprar_num"]  # Variable objetivo (target)

#separar train y test
X_train,X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inicializar el preprocesador con ColumnTransformer integrando las variables ordenadas
preprocesador = ColumnTransformer(
    transformers=[
        ('precio', OrdinalEncoder(categories=orden_precio), ['Precio']),
        ('kilometraje', OrdinalEncoder(categories=orden_kilometraje), ['Kilometraje']),
        ('antiguedad', OrdinalEncoder(categories=orden_antiguedad), ['Antiguedad']),
        ('marca', OrdinalEncoder(categories=orden_marca), ['Marca']),
        ('binarias', OrdinalEncoder(categories=orden_binarias), ['Historial_Accidentes', 'Financiamiento_Disponible'])
    ],
    remainder='passthrough'
)

# 2. Instanciar el modelo base
rf_base = RandomForestClassifier(random_state=42)

# Crear el Pipeline uniendo el preprocesador y el modelo base
pipeline = Pipeline(steps=[
    ('preprocesamiento', preprocesador),
    ('rf_base', rf_base)
])

# 1. Definir el espacio de búsqueda (Parameter Grid)
# Se añade el prefijo 'rf_base__' para que GridSearchCV entienda que van al RandomForest dentro del pipeline
param_grid = {
    'rf_base__n_estimators': [100, 300, 500],       # Cantidad de árboles
    'rf_base__max_depth': [8, 12, 15, None],        # Profundidad de cada árbol
    'rf_base__min_samples_split': [5, 10, 20]       # Resistencia al sobreajuste
}

# 3. Configurar GridSearchCV enfocado en la métrica principal (Recall)
grid_search = GridSearchCV(
    estimator=pipeline,    # Pasamos el pipeline completo en lugar del modelo base
    param_grid=param_grid,
    scoring='f1',    # El modelo ganador será el que tenga mejor F1-Score
    cv=5,                  # Validación cruzada de 5 particiones
    n_jobs=-1,             # Usa todos los núcleos del procesador
    verbose=2              # Muestra el progreso en la consola
)

# 4. Ejecutar la búsqueda con los datos de entrenamiento
print("Iniciando Grid Search. Esto evaluará 36 combinaciones x 5 pliegues = 180 ajustes...")
grid_search.fit(X_train, y_train)

# 5. Extraer y mostrar al ganador
print(f"\nMejores hiperparámetros encontrados para maximizar F1-Score:")
print(grid_search.best_params_)

# 6. Generar predicciones con el mejor modelo
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test)


def evaluar_spaceship_model(y_test, y_pred, modelo_nombre="Modelo"):
    print(f"--- Evaluación de: {modelo_nombre} ---")

    # 1. Calcular la Matriz de Confusión para extraer TN, FP, FN, TP
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 2. Calcular las métricas directas
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # 3. Calcular Specificity (Tasa de Verdaderos Negativos)
    specificity = tn / (tn + fp)

    # 4. Reporte alineado a tus prioridades
    print(f"Métrica Principal (Criterio de decisión):")
    print(f" * F1-Score:    {f1:.4f}")

    print(f"\nMétricas de Referencia / Contexto:")
    print(f" * Accuracy:    {accuracy:.4f}")
    print(f" * Recall:      {recall:.4f}")
    print(f" * Precision:   {precision:.4f}")
    print(f" * Specificity: {specificity:.4f}")

    # 5. Visualizar el trade-off con la Matriz de Confusión
    print("\nGenerando Matriz de Confusión...")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No comprar (0)", "Comprar (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión - {modelo_nombre}")
    plt.show()

evaluar_spaceship_model(y_test, y_pred_best_rf, "Random Forest para autos")