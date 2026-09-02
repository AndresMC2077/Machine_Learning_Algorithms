import numpy as np
import pandas as pd
#solo para las metricas
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay

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
def plot_tree_graph(tree):
    """Convierte el diccionario ID3 en un grafo y lo muestra en pantalla."""
    G = nx.DiGraph()
    node_id = 0
    labels = {}
    edge_labels = {}

    # Función recursiva para mapear el diccionario a NetworkX
    def traverse(node, parent_id=None, branch_val=None, x=0, y=0, layer_width=1.0):
        nonlocal node_id
        current_id = node_id
        node_id += 1

        # Identificar si es nodo u hoja para la etiqueta
        if node["type"] == "leaf":
            labels[current_id] = f"Hoja:\n{node['label']}"
            color = "lightgreen"
        else:
            labels[current_id] = f"¿{node['feature']}?"
            color = "lightblue"
            
        # Agregamos el nodo con sus coordenadas (x, y)
        G.add_node(current_id, pos=(x, y), color=color)

        # Conectar con el padre si existe
        if parent_id is not None:
            G.add_edge(parent_id, current_id)
            edge_labels[(parent_id, current_id)] = str(branch_val)

        # Recursividad para los hijos
        if node["type"] == "node":
            children = node["children"]
            n_children = len(children)
            if n_children > 0:
                # Matemáticas simples para que los nodos no se encimen
                dx = layer_width / n_children
                start_x = x - layer_width/2 + dx/2
                for i, (val, child_node) in enumerate(children.items()):
                    child_x = start_x + i * dx
                    child_y = y - 1 # Bajamos un nivel en Y
                    traverse(child_node, current_id, val, child_x, child_y, layer_width=dx*0.9)

    # Iniciamos el recorrido
    traverse(tree, x=0, y=0, layer_width=100)

    # Extraer atributos para dibujar
    pos = nx.get_node_attributes(G, 'pos')
    colors = [node[1]['color'] for node in G.nodes(data=True)]

    # Dibujar usando matplotlib
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, 
            node_size=3000, node_color=colors, 
            font_size=10, font_weight="bold", arrows=False, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Visualización del Árbol de Decisión ID3 (Custom)")
    # El plt.show() abre la ventana interactiva y NO guarda el archivo
    plt.show()

#evaluar con metricas de desempeño, solo se usa sklearn en este paso
def evaluate_tree(tree, df_test, target):
    """Calcula métricas y muestra la matriz de confusión gráficamente."""
    y_true = df_test[target]
    y_pred = [predict(tree, row) for _, row in df_test.iterrows()]
    
    print("\n" + "="*40)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*40)
    print(f"Exactitud (Accuracy): {accuracy_score(y_true, y_pred):.4f}\n")
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # MAGIA GRÁFICA PARA LA MATRIZ DE CONFUSIÓN
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        cmap="Blues", 
        colorbar=True
    )
    disp.ax_.set_title("Matriz de Confusión")
    plt.show() # Esto abre la gráfica de la matriz
    
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
plot_tree_graph(tree)
print("\nEvaluación del modelo:")
evaluate_tree(tree, df_test, target)

#conjuntos con los que se probó el modelo.
print("\nConjunto de entrenamiento: ",df_train)
print("\nConjunto de prueba: ",df_test)



