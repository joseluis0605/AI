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
        subset_entropy = entropy([row['T'] for row in subset])
        probability = len(subset) / total
        attribute_entropy += probability * subset_entropy
    return entropy([row['T'] for row in data]) - attribute_entropy

def majority_class(data):
    counter = Counter(data)
    majority = counter.most_common(1)[0][0]
    return majority

def build_tree(data, attributes):
    classes = [row['T'] for row in data]
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
        tree[best_attribute][value] = build_tree(subset, remaining_attributes)
    return tree

def visualize_tree(tree, feature_names, class_names):
    dot = graphviz.Digraph()
    add_nodes(tree, dot, feature_names, class_names)
    dot.render('decision_tree.png', format='png', cleanup=True)

def add_nodes(tree, dot, feature_names, class_names, parent_node=None, edge_label=None):
    if isinstance(tree, str):
        dot.node(tree, label=tree, shape='box')
        if parent_node is not None:
            dot.edge(parent_node, tree, label=edge_label)
    else:
        attribute = next(iter(tree))
        if parent_node is not None:
            dot.node(attribute, label=attribute, shape='box')
            dot.edge(parent_node, attribute, label=edge_label)
        for value, subtree in tree[attribute].items():
            if isinstance(subtree, str):
                dot.node(subtree, label=subtree, shape='box')
                dot.edge(attribute, subtree, label=value)
            else:
                child_node = attribute + '_' + str(value)
                dot.node(child_node, label=value, shape='box')
                dot.edge(attribute, child_node, label=value)
                add_nodes(subtree, dot, feature_names, class_names, parent_node=child_node, edge_label=value)

# Ejemplo de uso
data = [
    {'x1': '1', 'x2': '0', 'x3': '1', 'x4': '0', 'x5': '0', 'x6': '0', 'T': '1'},
    {'x1': '1', 'x2': '0', 'x3': '1', 'x4': '1', 'x5': '0', 'x6': '0', 'T': '1'},
    {'x1': '1', 'x2': '0', 'x3': '1', 'x4': '0', 'x5': '1', 'x6': '0', 'T': '1'},
    {'x1': '1', 'x2': '1', 'x3': '0', 'x4': '0', 'x5': '1', 'x6': '1', 'T': '1'},
    {'x1': '1', 'x2': '1', 'x3': '1', 'x4': '1', 'x5': '0', 'x6': '0', 'T': '1'},
    {'x1': '1', 'x2': '0', 'x3': '0', 'x4': '0', 'x5': '1', 'x6': '1', 'T': '1'},
    {'x1': '1', 'x2': '0', 'x3': '0', 'x4': '0', 'x5': '1', 'x6': '0', 'T': '0'},
    {'x1': '0', 'x2': '1', 'x3': '1', 'x4': '1', 'x5': '0', 'x6': '1', 'T': '1'},
    {'x1': '0', 'x2': '1', 'x3': '1', 'x4': '0', 'x5': '1', 'x6': '1', 'T': '0'},
    {'x1': '0', 'x2': '0', 'x3': '0', 'x4': '1', 'x5': '1', 'x6': '0', 'T': '0'},
    {'x1': '0', 'x2': '1', 'x3': '0', 'x4': '1', 'x5': '0', 'x6': '1', 'T': '0'},
    {'x1': '0', 'x2': '0', 'x3': '0', 'x4': '1', 'x5': '0', 'x6': '1', 'T': '0'},
    {'x1': '0', 'x2': '1', 'x3': '1', 'x4': '0', 'x5': '1', 'x6': '1', 'T': '0'},
    {'x1': '0', 'x2': '1', 'x3': '1', 'x4': '1', 'x5': '0', 'x6': '0', 'T': '0'}
]

attributes = ['x1', 'x2', 'x3', 'x4','x5','x6']

tree = build_tree(data, attributes)

feature_names = attributes
class_names = ['0', '1']

visualize_tree(tree, feature_names, class_names)