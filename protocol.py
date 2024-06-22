from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from Utils import Utils
import pickle


selected_metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
}

selected_classifiers = {
    "RFC_20": RandomForestClassifier(max_depth=20, random_state=0),
    "DTC": DecisionTreeClassifier(random_state=0)
}

n_splits = 5
n_repeats = 1  # as StratifiedKFold is used
skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)

def getAllResults(X, y, selected_classifiers, selected_metrics, filename="results"):
    all_results = {classifier: {metric: [] for metric in selected_metrics} for classifier in selected_classifiers}
    unique_labels = set(np.unique(y))
    # for i, (train, test) in enumerate(skf.split(X, y)):
    for i, (X_train, X_test, X_dsel, y_train, y_test, y_dsel) in enumerate(dsel_rskf(X, y, rng)):
        print(f"Iteration {i} started.")
        for classifier in selected_classifiers:
            # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            print(f"    Classifier={classifier}.")
            clf = selected_classifiers[classifier]
            clf.fit(X_train, y_train.ravel())
            y_pred = clf.predict(X_test)
            Utils.save_confusion_matrix(y_test, y_pred, unique_labels, experimentName=f"{classifier}_{i}")

            for metric in selected_metrics:
                result = selected_metrics[metric](y_test.ravel(), y_pred)
                all_results[classifier][metric].append(result)
                print(f"    Classifier passed. Results: {result}")
            print(f"Iteration {i} passed.")

    print(all_results)
    Utils.print_aggregated_cv_results(all_results)

    clfs_len = len(selected_classifiers)
    for selected_metric in selected_metrics:
        print(selected_metric)
        scores = np.zeros((clfs_len, n_splits * n_repeats))

        for idx, classifier in enumerate(all_results):
            for metric in all_results[classifier]:
                if metric == selected_metric:
                    scores[idx] = all_results[classifier][metric]

        Utils.compare_classifiers(scores, clfs_len, selected_classifiers)

        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        return all_results
