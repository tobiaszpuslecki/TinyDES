import ctypes
import math
import os
from keras.datasets import mnist, fashion_mnist
from deslib.des.des_clustering import DESClustering
from deslib.static import (StackedClassifier, SingleBest, StaticSelection, Oracle)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from deslib.des import KNORAE, KNORAU
from sklearn.preprocessing import StandardScaler
from micromlgen import port
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from Utils import Utils
import pickle


def load_dataset(rng_, get_whole=True, std_scaler=False, ds="MNIST"):
    # X_train, X_test, y_train, y_test = mnist.train_images(), mnist.test_images(), mnist.train_labels(), mnist.test_labels()
    if ds == "MNIST":
        (X_train_, y_train_), (X_test_, y_test_) = mnist.load_data()
    else:
        (X_train_, y_train_), (X_test_, y_test_) = fashion_mnist.load_data()
    X_ = np.concatenate((X_train_, X_test_), axis=0)
    y_ = np.concatenate((y_train_, y_test_), axis=0).ravel()
    # Reshape the 3D array to a 2D array - flatten
    X_ = X_.reshape(70000, 28 * 28)
    # split the data into training and test data
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.33, random_state=rng_)
    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train_ = scalar.fit_transform(X_train_)
    X_test_ = scalar.transform(X_test_)
    # Split the data into training and DSEL for DS techniques
    X_train_, X_dsel_, y_train_, y_dsel_ = train_test_split(X_train_, y_train_, test_size=0.5, random_state=rng_)

    if get_whole:
        if std_scaler:
            scalar = StandardScaler()
            X_ = scalar.fit_transform(X_)

        return X_, y_
    else:
        return X_dsel_, X_test_, X_train_, y_dsel_, y_test_, y_train_


def plot_accs(scores_, names_, selected_metric_, ds):
    def addlabels(x, y):
        for i in range(len(x)):
            plt.text(i, 0.5, y[i], ha='center')

    number = scores_.shape[0]
    fig, ax = plt.subplots(figsize=(number, 8))
    # fig, ax = plt.subplots()
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan']
    ax.bar(names_, scores_, color=bar_colors[:number], edgecolor='k')
    ax.set_ylabel(f'{selected_metric_} on the test set (%)', fontsize=13)
    ax.set_title(f'{selected_metric_} metric depending on selected method for {ds}')
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    plt.subplots_adjust(bottom=0.18)
    addlabels(names_, np.round(scores_, decimals=3))
    plt.show()


def cpredict(x_, desclustering_):
    def compute_euclidean_distance(x, y_):
        return np.sqrt(np.sum((x - y_) ** 2))

    cluster_centers = desclustering_.clustering_.cluster_centers_
    indices = desclustering_.indices_

    np.set_printoptions(threshold=np.inf, suppress=True, precision=6)
    # print(f"Cluster: {repr(cluster_centers)}")
    # print(f"indices: {repr(indices)}")
    # print(f"Cluster size: {cluster_centers.shape}")

    # calculate distance between X and all cluster centers
    euclidean_distances = []
    for cc in cluster_centers:
        ed = compute_euclidean_distance(x_, cc)
        euclidean_distances.append(ed)
    # get the closest cluster
    cluster_index = np.argmin(euclidean_distances)

    selected_classifiers_idx = indices[cluster_index]

    used_pool = pool_classifiers.estimators_

    from statistics import mode
    votes = []
    # calculate clfs from this cluster
    for idx_ in selected_classifiers_idx:
        clf_ = used_pool[idx_]
        pred = clf_.predict(x_)
        votes.append(pred[0])
    votes = list(votes)
    # majority vote using clfs
    # return predicted label
    return mode(votes)


def convert_to_c(idx_, pc_, desclustering_, j_value, cluster_no=5, probe_res=784):
    indices = desclustering_.indices_
    # not all clfs from pool will be in indices
    used_clfs = set(indices.flatten())

    with open(f"Trees.h", 'w') as file:
        # begin header guard
        header = f"#ifndef TREES_H\n" + f"#define TREES_H\n" + f"#define PROBE_RES {probe_res}\n"
        header += f"#define CLUSTERS_NO {cluster_no}\n"
        header += f"#define J_VALUE {j_value}\n"
        file.write(header)

        cluster_centers = desclustering_.clustering_.cluster_centers_

        np.set_printoptions(threshold=np.inf, suppress=True, precision=6)

        cluster_centers = repr(cluster_centers)
        cluster_centers = cluster_centers.replace("array(", "")
        cluster_centers = cluster_centers.replace(")", ";")
        cluster_centers = cluster_centers.replace("]", "}")
        cluster_centers = cluster_centers.replace("[", "{")
        cluster_centers = f"\nfloat cluster_centers[CLUSTERS_NO][PROBE_RES] =" + cluster_centers
        file.write(cluster_centers)

        indices = repr(indices)
        indices = indices.replace("array(", "")
        indices = indices.replace(")", ";")
        indices = indices.replace("]", "}")
        indices = indices.replace("[", "{")
        indices = f"\nuint8_t indices[CLUSTERS_NO][J_VALUE] =" + indices
        file.write(indices)

        # all trees
        for idx_, tree in enumerate(pc_.estimators_):
            if idx_ not in used_clfs:
                continue
            string = port(tree)
            # remove micromlgen specific begin
            string = string[string.find('int'):]
            # number of predict function
            fun_name = "predict" + str(idx_)
            string = string.replace("predict", fun_name)
            string = string.replace("int", "\nuint8_t")
            # remove micromlgen specific end
            string = string[:string.find('protected')]
            file.write(string)

        # array of function pointers
        ptrs = "\nuint8_t(*func_ptr[])(float *x) = {"
        # as we get classifiers from pointers array it is required
        # to keep proper array size (zero is kind of padding here)
        for idx_ in range(45):
            if idx_ in used_clfs:
                ptrs += f"predict{idx_}, "
            else:
                ptrs += f"0, "
        # rm last space and colon
        ptrs = ptrs[:-2]
        ptrs += "};\n\n"
        file.write(ptrs)

        # end header guard
        header = f"\n#endif // TREES_H \n"
        file.write(header)

    os.system(f"cp Trees.h data/Trees{idx_}.h")


def ccpredict(x_):
    so_file = "my_functions.so"
    my_functions = ctypes.CDLL(so_file)

    my_functions.cpredict.argstype = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    my_functions.cpredict.restype = ctypes.c_uint8

    input_ = x_[0]

    return my_functions.cpredict((ctypes.c_float * len(input_))(*input_), len(input_))


def rskf(X_, y_, rng_, n_repeats_=5, n_splits_=2):
    rskf_ = RepeatedStratifiedKFold(n_splits=n_splits_, n_repeats=n_repeats_, random_state=rng_)
    for train_index, test_index in rskf_.split(X, y):
        yield X_[train_index], X_[test_index], y_[train_index], y_[test_index]


def dsel_rskf(X_, y_, rng_, n_repeats_=5, n_splits_=2, test_size_=0.5):
    for X_train_, X_test_, y_train_, y_test_ in rskf(X_, y_, rng_, n_repeats_, n_splits_):
        X_train_, X_dsel_, y_train_, y_dsel_ = train_test_split(X_train_, y_train_, test_size=test_size_,
                                                                random_state=rng_)
        yield X_train_, X_test_, X_dsel_, y_train_, y_test_, y_dsel_


# if __name__ == '__main__':
#     rng = np.random.RandomState(42)
#     X, y = load_dataset(rng_=rng, get_whole=True, std_scaler=False)
#     all_scores = []
#
#     for idx, (X_train, X_test, X_dsel, y_train, y_test, y_dsel) in enumerate(dsel_rskf(X, y, rng)):
#
#         print("---" * 20)
#
#         max_depth, n_estimators, max_depth_2, n_estimators_2 = 10, 25, 5, 20  # RF 0.92115
#         print(max_depth, n_estimators, max_depth_2, n_estimators_2)
#
#         pc = RandomForestClassifier(random_state=rng, n_estimators=n_estimators, max_depth=max_depth)
#         pc.fit(X_train, y_train)
#         # combine with another pool
#         pc_2 = RandomForestClassifier(random_state=rng, n_estimators=n_estimators_2, max_depth=max_depth_2)
#         pc_2.fit(X_train, y_train)
#         pc.estimators_ += pc_2.estimators_
#         pc.n_estimators = len(pc.estimators_)
#
#         n_estimators = len(pc.estimators_)
#         pool_classifiers = pc
#
#         """
#         DES
#         """
#         model_voting = VotingClassifier(estimators=[("rf", pc)]).fit(X_train, y_train)
#         stacked = StackedClassifier(pool_classifiers, random_state=rng)
#         static_selection = StaticSelection(pool_classifiers, random_state=rng)
#         single_best = SingleBest(pool_classifiers, random_state=rng)
#         knorau = KNORAU(pool_classifiers, random_state=rng)
#         kne = KNORAE(pool_classifiers, random_state=rng)
#         oracle = Oracle(pool_classifiers).fit(X_train, y_train)
#
#         pct_diversity = 0.33
#         k_ = KMeans(n_clusters=5, random_state=rng)
#         desclustering = DESClustering(pool_classifiers, random_state=rng, pct_diversity=pct_diversity, clustering=k_)
#
#
#
#         methods = [
#             ('Single Best    ', single_best),
#             ('Static Selection', static_selection),
#             # ('Stacked', stacked),
#             ('Voting', model_voting),
#             ('KNORA-U', knorau),
#             ('KNORA-E', kne),
#             ('DES-Clustrering', desclustering),
#             ('Oracle', oracle)
#         ]
#
#         # names = ['Single Best', 'Static Selection', 'Stacked', 'Voting', 'KNORA-U', 'KNORA-E', 'DES-Clustrering', 'Oracle']
#         names = ['Single Best', 'Static Selection', 'Voting', 'KNORA-U', 'KNORA-E', 'DES-Clustrering', 'Oracle']
#         # names = ['Single Best', 'DES-Clustrering']
#
#         # Fit the DS techniques
#         scores = []
#         for name, method in methods:
#             method.fit(X_dsel, y_dsel)
#             score = method.score(X_test, y_test)
#             scores.append(score)
#             print(f"Classification accuracy {name} = {score:.4f}")
#
#         J_ = desclustering.J_
#         print(f"J={J_}")
#         convert_to_c(idx, pc, desclustering, J_)
#         print(f'File Size in kBytes is {os.stat("Trees.h").st_size / 1000}')
#         # call c compiler
#         os.system(f"cc -fPIC -shared -o my_functions.so my_functions.c")
#
#         # y_pred = [cpredict(x.reshape(1, -1), desclustering) for x in X_test]
#         y_pred = [ccpredict(x.reshape(1, -1)) for x in X_test]
#         print(f"custom acc = {accuracy_score(y_test, y_pred):.4f}")
#
#         all_scores.append(scores)
#
#     all_scores = np.array(all_scores)
#     mean_score, std_score = all_scores.mean(axis=0), all_scores.std(axis=0)
#
#     for name, mean, std in zip(names, mean_score, std_score):
#         print(f"{name} - Overall Accuracy (std): {mean:.3f} ({std:.3f})")
#
# plot_accs(mean_score, names)


def get_pool(X_train_, y_train_):
    max_depth, n_estimators, max_depth_2, n_estimators_2 = 10, 25, 5, 20  # RF 0.92115
    # print(max_depth, n_estimators, max_depth_2, n_estimators_2)
    pc = RandomForestClassifier(random_state=rng, n_estimators=n_estimators, max_depth=max_depth)
    pc.fit(X_train_, y_train_)
    # combine with another pool
    pc_2 = RandomForestClassifier(random_state=rng, n_estimators=n_estimators_2, max_depth=max_depth_2)
    pc_2.fit(X_train_, y_train_)
    pc.estimators_ += pc_2.estimators_
    pc.n_estimators = len(pc.estimators_)
    return pc


if __name__ == '__main__':

    ds = "Fashion MNIST"

    k = 5
    n_estimators = 45
    n_splits, n_repeats = 5, 2
    rng = np.random.RandomState(42)
    X, y = load_dataset(rng_=rng, get_whole=True, std_scaler=False, ds=ds)

    selected_metrics = {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        # "F1 Score Weighted": lambda y_true_, y_pred_: f1_score(y_true_, y_pred_, average='weighted'),
        "F1 Score Micro": lambda y_true_, y_pred_: f1_score(y_true_, y_pred_, average='micro'),
        "F1 Score Macro": lambda y_true_, y_pred_: f1_score(y_true_, y_pred_, average='macro'),
        #
        # "Precision Score Weighted": lambda y_true_, y_pred_: precision_score(y_true_, y_pred_, average='weighted'),
        # "Precision Score Micro": lambda y_true_, y_pred_: precision_score(y_true_, y_pred_, average='micro'),
        # "Precision Score Macro": lambda y_true_, y_pred_: precision_score(y_true_, y_pred_, average='macro'),
        #
        # "Recall Score Weighted": lambda y_true_, y_pred_: recall_score(y_true_, y_pred_, average='weighted'),
        # "Recall Score Micro": lambda y_true_, y_pred_: recall_score(y_true_, y_pred_, average='micro'),
        # "Recall Score Macro": lambda y_true_, y_pred_: recall_score(y_true_, y_pred_, average='macro'),
    }

    selected_classifiers = {
        # "Single Best": lambda pool_classifiers_: SingleBest(pool_classifiers_, random_state=rng),
        # "Static Selection": lambda pool_classifiers_: StaticSelection(pool_classifiers_, random_state=rng),
        "KNORA-U": lambda pool_classifiers_: KNORAU(pool_classifiers_, random_state=rng),
        # "KNORA-E": lambda pool_classifiers_: KNORAE(pool_classifiers_, random_state=rng),
        # "DES-Clustering_5": lambda pool_classifiers_: DESClustering(pool_classifiers_, random_state=rng, pct_diversity=0.11),
        # "DES-Clustering_10": lambda pool_classifiers_: DESClustering(pool_classifiers_, random_state=rng, pct_diversity=0.22),
        # "DES-Clustering_15": lambda pool_classifiers_: DESClustering(pool_classifiers_, random_state=rng, pct_diversity=0.33),
        "DES-Clustering_20": lambda pool_classifiers_: DESClustering(pool_classifiers_, random_state=rng, pct_diversity=0.44),
        # "Oracle": lambda pool_classifiers_: Oracle(pool_classifiers_),
    }

    all_results = {classifier: {metric: [] for metric in selected_metrics} for classifier in selected_classifiers}

    # J = int(math.ceil(n_estimators * pct_diversity)) # n_estimators has to be large to have bigger clfs choice
    # print(f"J value is {J} for pct_diversity={pct_diversity}")

    filename = f"results"
    names = [name for name in selected_classifiers]

    unique_labels = set(np.unique(y))
    for i, (X_train, X_test, X_dsel, y_train, y_test, y_dsel) in enumerate(dsel_rskf(X, y, rng)):
        for classifier in selected_classifiers:
            # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
            print(f"\n({i}) Classifier={classifier}.")
            pool_classifiers = get_pool(X_train, y_train)
            clf = selected_classifiers[classifier](pool_classifiers)
            clf.fit(X_train, y_train.ravel())
            if "Oracle" not in classifier:
                y_pred = clf.predict(X_test)
            else:
                y_pred = clf.predict(X_test, y_test)
            Utils.save_confusion_matrix(y_test, y_pred, unique_labels, experimentName=f"{classifier}_{i}")

            for metric in selected_metrics:
                result = selected_metrics[metric](y_test.ravel(), y_pred)
                all_results[classifier][metric].append(result)
                print(f"    Results {metric}: {result}")

            if "DES-Clustering" in classifier:
                convert_to_c(i, pool_classifiers, clf, clf.J_, k)
                os.system(f"cc -fPIC -shared -o my_functions.so my_functions.c")
                y_pred = [ccpredict(x.reshape(1, -1)) for x in X_test]
                print(f"custom acc = {accuracy_score(y_test, y_pred):.4f}")

    # print(all_results)
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

    with open(f'{filename}.pkl', 'rb') as fp:
        all_results = pickle.load(fp)

    for selected_metric in selected_metrics:
        # print(selected_metric)
        scores = np.zeros((len(selected_classifiers), n_splits * n_repeats))

        for idx, classifier in enumerate(all_results):
            for metric in all_results[classifier]:
                if metric == selected_metric:
                    scores[idx] = all_results[classifier][metric]

        mean_score, std_score = scores.mean(axis=1), scores.std(axis=1)
        plot_accs(mean_score, names, selected_metric, ds)
        # for idxs, name in enumerate(names):
        #     print(f"{name} - {selected_metric}={mean_score[idxs]} ({std_score[idxs]})")
