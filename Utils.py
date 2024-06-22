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
