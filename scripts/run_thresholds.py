import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from src.models.ensemble import load_fold_models, EnsembleWrapper


def run_thresholds(X_train_path, X_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    X_full = pd.concat([X_train, X_test], ignore_index=True)

    models = load_fold_models()
    ensemble = EnsembleWrapper(models)

    y_surrogate = ensemble.predict(X_full)[:, 1]

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)
    tree.fit(X_full, y_surrogate)

    rules = export_text(tree, feature_names=list(X_full.columns))
    print("\nSurrogate Tree Rules:\n")
    print(rules)

    thresholds = []
    for node, feat_idx in enumerate(tree.tree_.feature):
        if feat_idx != -2:
            thresholds.append({
                "feature": X_full.columns[feat_idx],
                "threshold": tree.tree_.threshold[node]
            })

    pd.DataFrame(thresholds).to_csv("surrogate_thresholds.csv", index=False)
