import math
from collections import Counter
import graphviz

def entropy(data):
    total = len(data)
    counter = Counter(data)
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy

def information_gain(data, attribute):
    total = len(data)
    attribute_values = set([row[attribute] for row in data])
    attribute_entropy = 0.0
    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_entropy = entropy([row['EsVenenosa'] for row in subset])
        probability = len(subset) / total
        attribute_entropy += probability * subset_entropy
    return entropy([row['EsVenenosa'] for row in data]) - attribute_entropy

def majority_class(data):
    counter = Counter(data)
    majority = counter.most_common(1)[0][0]
    return majority

def build_tree(data, attributes, parent_attribute=None, parent_value=None):
    classes = [row['EsVenenosa'] for row in data]
    if len(set(classes)) == 1:
        return classes[0]
    if len(attributes) == 0:
        return majority_class(classes)
    best_attribute = max(attributes, key=lambda attr: information_gain(data, attr))
    tree = {best_attribute: {}}
    attribute_values = set([row[best_attribute] for row in data])
    for value in attribute_values:
        subset = [row for row in data if row[best_attribute] == value]
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        subtree = build_tree(subset, remaining_attributes, best_attribute, value)
        tree[best_attribute][value] = subtree
    if parent_attribute is not None:
        if parent_value is not None:
            tree = {parent_attribute + '_' + str(parent_value): tree}
        else:
            tree = {parent_attribute: tree}
    return tree

def visualize_tree(tree, feature_names, class_names):
    dot = graphviz.Digraph()
    add_nodes(tree, dot, feature_names, class_names)
    dot.render('decision_treev2.png', format='png', cleanup=True)

def add_nodes(tree, dot, feature_names, class_names, parent_node=None, edge_label=None):
    if isinstance(tree, str):
        dot.node(tree, label=tree, shape='box')
        if parent_node is not None:
            dot.edge(parent_node, tree, label=edge_label)
    else:
        attribute = next(iter(tree))
        if parent_node is not None:
            intermediate_node = attribute + '_' + str(id(tree))
            dot.node(intermediate_node, label=attribute, shape='box')
            dot.edge(parent_node, intermediate_node, label=edge_label)
            parent_node = intermediate_node
        if isinstance(tree[attribute], str):  # Comprueba si el nodo actual es una hoja
            dot.node(tree[attribute], label=tree[attribute], shape='box')
            dot.edge(parent_node, tree[attribute], label=edge_label)
        else:
            for value, subtree in tree[attribute].items():
                child_node = attribute + '_' + str(value)
                dot.node(child_node, label=value, shape='box')
                dot.edge(attribute, child_node, label=value)
                add_nodes(subtree, dot, feature_names, class_names, parent_node=child_node, edge_label=value)


# Ejemplo de uso
# Ejemplo de uso
data = [
    {'PesaMucho': 'NO', 'EsMaloliente': 'NO', 'EsConManchas': 'NO', 'EsSuave': 'NO', 'EsVenenosa': 'NO'},
    {'PesaMucho': 'NO', 'EsMaloliente': 'NO', 'EsConManchas': 'SI', 'EsSuave': 'NO', 'EsVenenosa': 'NO'},
    {'PesaMucho': 'SI', 'EsMaloliente': 'SI', 'EsConManchas': 'NO', 'EsSuave': 'SI', 'EsVenenosa': 'NO'},
    {'PesaMucho': 'SI', 'EsMaloliente': 'NO', 'EsConManchas': 'NO', 'EsSuave': 'SI', 'EsVenenosa': 'SI'},
    {'PesaMucho': 'NO', 'EsMaloliente': 'SI', 'EsConManchas': 'SI', 'EsSuave': 'NO', 'EsVenenosa': 'SI'},
    {'PesaMucho': 'NO', 'EsMaloliente': 'NO', 'EsConManchas': 'SI', 'EsSuave': 'SI', 'EsVenenosa': 'SI'},
    {'PesaMucho': 'NO', 'EsMaloliente': 'NO', 'EsConManchas': 'NO', 'EsSuave': 'SI', 'EsVenenosa': 'SI'},
    {'PesaMucho': 'SI', 'EsMaloliente': 'SI', 'EsConManchas': 'NO', 'EsSuave': 'NO', 'EsVenenosa': 'SI'}
]

attributes = ['PesaMucho', 'EsMaloliente', 'EsConManchas', 'EsSuave']

tree = build_tree(data, attributes)

feature_names = attributes
class_names = ['No', 'Sí']

visualize_tree(tree, feature_names, class_names)