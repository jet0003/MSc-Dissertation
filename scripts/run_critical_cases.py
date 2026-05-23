import pandas as pd
from src.analysis.critical_cases import select_critical_cases, nearest_neighbour_pairs


def run(preds_path, X_path):
    preds = pd.read_csv(preds_path)
    X = pd.read_csv(X_path)

    selected = select_critical_cases(preds)
    print("Critical patient IDs:", selected)

    nn_pairs = nearest_neighbour_pairs(X, preds)
    print("Nearest neighbour pairs:", nn_pairs)
