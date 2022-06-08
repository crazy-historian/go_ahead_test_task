import argparse
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Decision Tree Regressor."
)
parser.add_argument(
    'filename', type=str, help='Filename with numeric data stored in tuples (x, y).'
)


def find_parameters(X: np.ndarray, y: np.ndarray) -> tuple:
    dec_tree_regression = DecisionTreeRegressor(max_depth=1, max_leaf_nodes=2, criterion='squared_error')
    dec_tree_regression.fit(X, y)

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = dec_tree_regression.predict(X_test)

    c = dec_tree_regression.tree_.threshold[0]
    a = dec_tree_regression.tree_.value[1][0]
    b = dec_tree_regression.tree_.value[2][0]

    text_representation = tree.export_text(dec_tree_regression, feature_names=['x'])
    print(text_representation)

    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="blue", label="data")
    plt.plot(X_test, y_1, color="red", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Дерево решений в применении к регрессии")
    plt.savefig('plot.png')

    return c, a, b


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename

    X = list()
    y = list()

    with open(filename, 'r') as file:
        for line in file:
            line = line.split(',')
            X.append(float(line[0]))
            y.append(float(line[1]))

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).ravel()

    c, a, b = find_parameters(X, y)

    print(f'c = {c}, a = {a[0]}, b = {b[0]}')
