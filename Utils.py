import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statistics import mean, stdev
import time
from sklearn.model_selection import GridSearchCV
from scipy import stats
import pickle

seed = 22
import random
import sklearn

random.seed(seed)
np.random.seed(seed)
sklearn.utils.check_random_state(22)

from scipy.stats import ttest_ind
from tabulate import tabulate


class Utils:

    # I put these part of work in Utils class temporarily. In future we should refactor whole repo before public release

    @staticmethod
    def plot_distribution(data, labels, title='Histogram', xlabel='Value', ylabel='Frequency', color='blue',
                          edgecolor='black', percentage=False):
        unique_labels, _ = np.unique(labels, return_counts=True)

        fig, ax = plt.subplots()

        weights = 100 * np.ones(len(data)) / len(data) if percentage else None

        _, bins, bars = ax.hist(data, bins=len(unique_labels), color=color, edgecolor=edgecolor, weights=weights)
        if percentage:
            ax.bar_label(bars, fmt=lambda x: f"{x:.1f}%")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # dirty hack...
        ax.set_xticks(bins[:-1] + (bins[1] - bins[0]) / 2)
        ax.set_xticklabels(unique_labels)

        return fig, ax

    @staticmethod
    def save_confusion_matrix(y_test, y_pred, unique_labels, experimentName, dir="cm"):
        if not os.path.exists(dir):
            os.makedirs(dir)

        cnf = confusion_matrix(y_test, y_pred)

        with open(f"{dir}/{experimentName}.pkl", "wb") as f:
            pickle.dump(cnf, f)

        plt.figure(figsize=(8, 6), dpi=70, facecolor='w', edgecolor='k')

        group_counts = ["{0:0.0f}".format(value) for value in cnf.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cnf.flatten() / np.sum(cnf)]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
        class_no = len(unique_labels)
        labels = np.asarray(labels).reshape(class_no, class_no)

        ax = sns.heatmap(cnf, cmap='Blues', annot=labels, fmt='', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(experimentName + ' Recognition')
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
        # plt.show()
        plt.savefig(f"{dir}/{experimentName}_cm.png")
        plt.clf()
        plt.cla()
        plt.close()

    @staticmethod
    def print_aggregated_cv_results(all_results):
        # resultDict = {metric: [] for metric in selected_metrics}
        for classifier in all_results:
            for metric in all_results[classifier]:
                results = all_results[classifier][metric]
                print(f"{classifier} - Overall Accuracy (std) for {metric}: {mean(results):.3f} ({stdev(results):.3f})")

    @staticmethod
    def print_aggregated_cv_results2(all_results):
        # resultDict = {metric: [] for metric in selected_metrics}
        for value in all_results:
            for classifier in all_results[value]:
                for metric in all_results[classifier]:
                    results = all_results[classifier][metric]
                    print(f"{classifier}_{value} - Overall Accuracy (std) for {metric}: {mean(results):.3f} ({stdev(results):.3f})")


    @staticmethod
    def tgrid_search(clf, train_images, train_labels, hyperparameters, cv_no):
        """
        This function should be used to test hyperparameters space and to find best params combination.
        Use it as desired ;)
        Example of use:

        cv_params = {
            'model__conv_layers1': [[(4,5,2,2)]],
            'model__dense_layers': [[50],[50,20]],
            'epochs':[3],
            'batch_size':[64]
        }
        cv_no = 5

        grid_result = tgrid_search(X, y, cv_params, cv_no)
        """

        start_time = time.time()
        grid = GridSearchCV(estimator=clf, param_grid=hyperparameters, cv=cv_no, scoring='accuracy', n_jobs=-1)
        grid_result = grid.fit(train_images, train_labels)
        print(f"Best accuracy: {grid_result.best_score_:.4f}, Best Params: {grid_result.best_params_}")
        print(f"--- {time.time() - start_time} seconds ---")
        return grid_result

    @staticmethod
    def compare_classifiers(scores, clfs_len, selected_classifiers, alpha=0.05, tablefmt="plain"):
        t_statistic = np.zeros((clfs_len, clfs_len))
        p_statistic = np.zeros((clfs_len, clfs_len))

        for i in range(clfs_len):
            for j in range(clfs_len):
                t_statistic[i][j], p_statistic[i][j] = ttest_ind(scores[i], scores[j])

        headers = [name for name in selected_classifiers]
        names_column = np.array([[name] for name in selected_classifiers])

        advantages, significances = np.zeros((clfs_len, clfs_len)), np.zeros((clfs_len, clfs_len))
        advantages[t_statistic > 0] = 1
        significances[p_statistic <= alpha] = 1
        stat_better = significances * advantages
        sig_bet_table = np.concatenate((names_column, stat_better), axis=1)
        sig_bet_table = tabulate(sig_bet_table, headers, tablefmt=tablefmt)
        print(f"\nStatistically significantly better:\n", sig_bet_table)

        print()
        for i in range(clfs_len):
            for j in range(clfs_len):
                if stat_better[i][j] >= 1:
                    print(f"{headers[i]} better than {headers[j]}") # or otherwise xD

    @staticmethod
    def balanced_accuracy_per_class(confusion_matrix):
        cm = np.array(confusion_matrix)

        num_classes = cm.shape[0]

        balanced_accuracies = []

        for i in range(num_classes):
            TP_i = cm[i, i]
            FN_i = np.sum(cm[i, :]) - TP_i
            FP_i = np.sum(cm[:, i]) - TP_i
            TN_i = np.sum(cm) - (TP_i + FN_i + FP_i)

            recall_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) != 0 else 0
            specificity_i = TN_i / (TN_i + FP_i) if (TN_i + FP_i) != 0 else 0

            balanced_accuracy_i = (recall_i + specificity_i) / 2
            balanced_accuracies.append(balanced_accuracy_i)

        return balanced_accuracies

    @staticmethod
    def f1_score_per_class(confusion_matrix):
        cm = np.array(confusion_matrix)

        num_classes = cm.shape[0]

        f1_scores = []

        for i in range(num_classes):
            TP_i = cm[i, i]
            FN_i = np.sum(cm[i, :]) - TP_i
            FP_i = np.sum(cm[:, i]) - TP_i

            precision_i = TP_i / (TP_i + FP_i) if (TP_i + FP_i) != 0 else 0
            recall_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) != 0 else 0

            if precision_i + recall_i != 0:
                f1_score_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
            else:
                f1_score_i = 0

            f1_scores.append(f1_score_i)

        return f1_scores

    @staticmethod
    def accuracy_per_class(confusion_matrix):
        cm = np.array(confusion_matrix)

        num_classes = cm.shape[0]

        accuracies = []

        for i in range(num_classes):
            TP_i = cm[i, i]
            FN_i = np.sum(cm[i, :]) - TP_i
            FP_i = np.sum(cm[:, i]) - TP_i
            TN_i = np.sum(cm) - (TP_i + FN_i + FP_i)

            accuracy_i = (TP_i + TN_i) / (TP_i + TN_i + FP_i + FN_i) if (TP_i + TN_i + FP_i + FN_i) != 0 else 0

            accuracies.append(accuracy_i)

        return accuracies
