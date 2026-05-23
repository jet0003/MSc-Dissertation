from lime.lime_tabular import LimeTabularExplainer


def build_lime_explainer(X_train, feature_names, class_names):
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
