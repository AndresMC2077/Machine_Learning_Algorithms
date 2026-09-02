import numpy as np
import pandas as pd
#solo para las metricas
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#función de entropía inicial de la variable y 
def entropy(y):
    _H = 0
    for label in np.unique(y):
        p = np.sum(y == label) / len(y)
        if p > 0:
            _H += p * np.log2(p)   
        else:
            _H += 0
    return -_H

#función de entropía condicional (cotejando con la variable y)
def entropy_2(y,x):
    _H = 0
    for label in np.unique(x):
        p = np.sum(x == label) / len(x)
        if p > 0:
            _H += p * entropy(y[x == label])   
        else:
            _H += 0
    return _H

def information_gain(y,x):
    return entropy(y) - entropy_2(y,x)

def majority_class(y):
    mode_val=y.value_counts().index[0]
    return mode_val

#examples: conjunto de datos
#target: variable objetivo
#atributes: conjunto de atributos
#parent_examples: conjunto de datos padre

def ID3(examples, target, attributes,parent_examples=None):
    #casos base
    #si está vacío, devuelve la clase mayoritaria del conjunto de datos padre osea, el nodo anterior
    if examples.empty:
            #devolver en forma de diccionario el tipo y la etiqueta
            return {"type": "leaf", "label": majority_class(parent_examples[target])}
    #si la clase es pura se devuelve el valor real de esa clase ej. comprar (si|no) y queremos el "si"
    elif entropy(examples[target])==0:
        return {"type": "leaf", "label": examples[target].iloc[0]}
    #si no hay más atributos devuelve la clase mayoritaria del conjunto actual
    elif len(attributes) == 0:
        return {"type": "leaf", "label":majority_class(examples[target])}

    
    else:
        #start
        best_gain=-1
        best_atribute=""
        #sacamos el atributo con mejor information gain
        for atribute in attributes:
            i_gain=information_gain(examples[target],examples[atribute])
            if best_gain<i_gain:
                best_gain=i_gain
                best_atribute=atribute
        #lo sacamos de la lista de atributos a revisar
        remaining=[attr for attr in attributes if attr!=best_atribute]
        #empezamos el arbolito con children como otro diccionario
        tree = {"type": "node", "feature": best_atribute, "children": {}}
        #para el diccionario de children
        #primero encontramos cada valor distinto en la columna de mejor atributo
        #luego filtramos con ese valor y se lo pasamos a la funcion recursiva, lo va a hacer para cada valor del atributo
        for b in np.unique(examples[best_atribute]):
            subset=examples[examples[best_atribute]==b]
            tree["children"][b] = ID3(subset,target,remaining,examples)

        return tree

#hacer una prediccion con el modelo "entrenado"
def predict(tree, row):
    #checo si es hoja, si si ya acabé
    if tree["type"] == "leaf":
        return tree["label"]
    #sino checo en que feature está y checo el valor de el registro a evaluar de esa feature
    else:
        feature=tree["feature"]
        value=row[feature]
        #si el valor está en que conocemos del entrenamiento hacemos la secursividad
        if value in tree["children"]:
            return predict(tree["children"][value],row)
        else:
            return "Unknown"    

#solo visualizar el arbol
def print_tree(tree, indent="", branch="[Raíz]"):
    """Imprime el árbol de forma recursiva con indentación."""
    if tree["type"] == "leaf":
        print(f"{indent} {branch} ---> Predicción: {tree['label']}")
        return

    feature = tree["feature"]
    print(f"{indent} {branch} ¿{feature}?")
    for value, child in tree["children"].items():
        print_tree(child, indent + "    ", f"Si es '{value}'")


#evaluar con metricas de desempeño, solo se usa sklearn en este paso
def evaluate_tree(tree, df_test, target):
    #Calcula métricas de clasificación para un dataset de prueba.
    y_true = df_test[target]
    # Hacemos la predicción fila por fila
    y_pred = [predict(tree, row) for _, row in df_test.iterrows()]
    
    print("\n" + "="*40)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*40)
    print(f"Exactitud (Accuracy): {accuracy_score(y_true, y_pred):.4f}\n")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_true, y_pred))
    print("\nReporte de Clasificación (Precision, Recall, F1):")
    print(classification_report(y_true, y_pred, zero_division=0))    


#construir el arbolito
df = pd.read_csv("dataset_compra_coche.csv")
df_train = df.sample(frac=0.7, random_state=42)
df_test = df.drop(df_train.index)
target = "Comprar"
features = [col for col in df.columns if col != target]
tree = ID3(df_train, target, features)
print("estructura del arbol:")
print_tree(tree)
print("\nEvaluación del modelo:")
evaluate_tree(tree, df_test, target)





