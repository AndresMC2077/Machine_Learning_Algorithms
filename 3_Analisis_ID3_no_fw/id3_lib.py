"""
id3_lib.py
-----------
Implementación del árbol de decisión ID3 (sin librerías de ML para el entrenamiento),
extraída de ID3_tree_mine.py para poder importarla desde otros scripts sin que
se ejecute el bloque de entrenamiento del archivo original.

Incluye:
- Clase ID3: entrenamiento (fit) del árbol.
- predict(tree, row): predicción de un registro.
- tree_depth(tree): profundidad real del árbol entrenado.
- tree_num_nodes(tree): número total de nodos (internos + hojas).
"""

import numpy as np
import pandas as pd


class ID3:
    def __init__(self, max_depth=None, min_samples_split=2):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth debe ser None o un entero mayor o igual a 0")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split debe ser un entero mayor o igual a 2")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y):
        entropy_value = 0
        for label in np.unique(y):
            probability = np.sum(y == label) / len(y)
            if probability > 0:
                entropy_value += probability * np.log2(probability)
        return -entropy_value

    def entropy_2(self, y, x):
        conditional_entropy = 0
        for label in np.unique(x):
            probability = np.sum(x == label) / len(x)
            if probability > 0:
                conditional_entropy += probability * self.entropy(y[x == label])
        return conditional_entropy

    def information_gain(self, y, x):
        return self.entropy(y) - self.entropy_2(y, x)

    def majority_class(self, y):
        return y.value_counts().index[0]

    def fit(self, examples, target, attributes=None):
        if attributes is None:
            attributes = [column for column in examples.columns if column != target]
        self.tree = self._build_tree(examples, target, attributes, depth=0)
        return self.tree

    def _build_tree(self, examples, target, attributes, parent_examples=None, depth=0):
        if examples.empty:
            return {"type": "leaf", "label": self.majority_class(parent_examples[target])}
        if self.entropy(examples[target]) == 0:
            return {"type": "leaf", "label": examples[target].iloc[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return {"type": "leaf", "label": self.majority_class(examples[target])}
        if len(examples) < self.min_samples_split:
            return {"type": "leaf", "label": self.majority_class(examples[target])}
        if len(attributes) == 0:
            return {"type": "leaf", "label": self.majority_class(examples[target])}

        best_attribute = max(
            attributes,
            key=lambda attribute: self.information_gain(examples[target], examples[attribute]),
        )
        remaining_attributes = [a for a in attributes if a != best_attribute]
        tree = {"type": "node", "feature": best_attribute, "children": {}}

        for value in np.unique(examples[best_attribute]):
            subset = examples[examples[best_attribute] == value]
            tree["children"][value] = self._build_tree(
                subset, target, remaining_attributes, examples, depth=depth + 1
            )
        return tree


def predict(tree, row):
    """Recorre el árbol para predecir la etiqueta de un registro (fila de un DataFrame)."""
    if tree["type"] == "leaf":
        return tree["label"]
    else:
        feature = tree["feature"]
        value = row[feature]
        if value in tree["children"]:
            return predict(tree["children"][value], row)
        else:
            # Combinación de atributo no vista durante el entrenamiento
            return "Unknown"


def tree_depth(tree):
    """Profundidad real del árbol (0 si es una sola hoja)."""
    if tree["type"] == "leaf":
        return 0
    return 1 + max(tree_depth(c) for c in tree["children"].values())


def tree_num_nodes(tree):
    """Número total de nodos del árbol (internos + hojas)."""
    if tree["type"] == "leaf":
        return 1
    return 1 + sum(tree_num_nodes(c) for c in tree["children"].values())
