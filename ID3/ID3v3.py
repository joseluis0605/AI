import numpy as np

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}

def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def split_dataset(dataset, attribute_index):
    unique_values = np.unique(dataset[:, attribute_index])
    subsets = []
    for value in unique_values:
        subset = dataset[dataset[:, attribute_index] == value]
        subsets.append(subset)
    return subsets

def choose_best_attribute(dataset, attributes):
    entropies = []
    for attribute_index in attributes:
        subsets = split_dataset(dataset, attribute_index)
        entropy_sum = 0
        for subset in subsets:
            labels = subset[:, -1]
            entropy = calculate_entropy(labels)
            weight = len(subset) / len(dataset)
            entropy_sum += weight * entropy
        entropies.append(entropy_sum)
    best_attribute_index = np.argmin(entropies)
    return attributes[best_attribute_index]

def create_decision_tree(dataset, attributes):
    labels = dataset[:, -1]
    unique_labels = np.unique(labels)

    # If all labels are the same, return a leaf node
    if len(unique_labels) == 1:
        return Node(label=unique_labels[0])

    # If there are no attributes left, return a leaf node with the majority label
    if len(attributes) == 0:
        majority_label = np.argmax(np.bincount(labels))
        return Node(label=majority_label)

    best_attribute = choose_best_attribute(dataset, attributes)
    node = Node(attribute=best_attribute)

    subsets = split_dataset(dataset, best_attribute)
    new_attributes = [attr for attr in attributes if attr != best_attribute]
    for subset in subsets:
        if len(subset) == 0:
            majority_label = np.argmax(np.bincount(labels))
            node.children[subset[0, best_attribute]] = Node(label=majority_label)
        else:
            node.children[subset[0, best_attribute]] = create_decision_tree(subset, new_attributes)

    return node

def print_decision_tree(node, indent=""):
    if node.label is not None:
        print(indent + "Leaf Node: " + str(node.label))
    else:
        print(indent + "Attribute: " + str(node.attribute))
        for value, child in node.children.items():
            print(indent + "Value " + str(value) + ":")
            print_decision_tree(child, indent + "  ")

# Define the dataset
dataset = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 0, 0, 1]
])

# Define the attribute indices
attributes = [0, 1, 2, 3]

# Build the decision tree
root = create_decision_tree(dataset, attributes)

# Print the decision tree
print_decision_tree(root)
