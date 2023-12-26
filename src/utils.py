import torch
import numpy as np
import pandas as pd
from global_var import *
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from collections import Counter
import matplotlib.pyplot as plt

def mse_each(x, x_rec):
    if type(x) == torch.Tensor:
        return torch.square(x - x_rec).mean(axis=1)
    else:
        return np.square(x - x_rec).mean(axis=1)

def TP(y_true, y_pred):
    return ((y_true != 0) & (y_pred != 0)).sum()

def FP(y_true, y_pred):
    return ((y_true == 0) & (y_pred != 0)).sum()

def TN(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum()

def FN(y_true, y_pred):
    return ((y_true != 0) & (y_pred == 0)).sum()

def save_result(result, file_name):
    df = pd.DataFrame(result, index=[0])
    target_file = os.path.join(RESULT_DIR, file_name + '.csv')
    if not os.path.exists(target_file):
        df.to_csv(target_file, index=False)
    else:
        df.to_csv(target_file, header=False, index=False, mode='a')

def inverse_norm(normalizer, dim, value):
    a = np.zeros((1, 30))
    a[0, dim] = value
    return normalizer.inverse_transform(a)[0, dim]

def norm_value(normalizer, dim, value):
    a = np.zeros((1, 30))
    a[0, dim] = value
    return normalizer.transform(a)[0, dim]   

# Print class distribution
def print_class_distribution(predicted_labels):
    class_counts = Counter(predicted_labels)
    total_samples = len(predicted_labels)
    distribution = {label: count / total_samples for label, count in class_counts.items()}
    print("Class distribution:")
    for label, percentage in distribution.items():
        print(f"Class {label}: {percentage * 100:.2f}%")

# Evaluate predictions (Simple)
def evaluate_predictions(y_test, predicted_labels, average=None):
    conf_matrix = confusion_matrix(y_test, predicted_labels)
    tn, fp, fn, tp = conf_matrix.ravel()
    tn, fp, fn, tp = tn/(tn+fp), fp/(tn+fp), fn/(fn+tp), tp/(fn+tp)
    # print("Confusion matrix:")
    # print(conf_matrix)
    # print("TP : ", tp)
    # print("FP : ", fp)
    # print("TN : ", tn)
    # print("FN : ", fn)

    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels, average=average)
    recall = recall_score(y_test, predicted_labels, average=average)
    f1 = f1_score(y_test, predicted_labels, average=average)

    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
    return tn, fp, fn, tp, accuracy, precision, recall, f1

# Evaluate_and_save_results
def evaluate_and_save_results_Hyperparameters( test_target, predictions, original_predictions, perturbed_predictions, dataset, baseline='None', black_model=None, max_level=5, n_beam=10, rho=0.3, eta=0.1):
    # Calculate classification metrics
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions, average='micro')
    recall = recall_score(test_target, predictions, average='micro')
    f1 = f1_score(test_target, predictions, average='micro')

    # Calculate evaluation metrics
    input_instances = len(test_target)

    # Completeness
    covered_by_rules = sum([1 for pred, true_label in zip(predictions, test_target) if pred == true_label and pred == 0])
    completeness = covered_by_rules / input_instances

    # Correctness
    correctly_classified = sum([1 for pred, true_label in zip(predictions, test_target) if pred == true_label])
    correctness = correctly_classified / input_instances

    # Fidelity
    consistent_predictions = np.sum(np.all([original_predictions == predictions], axis=0))
    fidelity = consistent_predictions / input_instances

    # Robustness
    def perturb_data_point(data_point, delta):
        return [x + delta for x in data_point]

    robustness_sum = 0
    for i,prediction in enumerate(predictions):
        if prediction == perturbed_predictions[i]:
            robustness_sum += 1

    robustness = robustness_sum / input_instances

    # Print confusion_matrix results
    conf_matrix = confusion_matrix(test_target, predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    tn, fp, fn, tp = np.resize(conf_matrix.ravel(), 4)
    tn, fp, fn, tp = tn/(tn+fp), fp/(tn+fp), fn/(fn+tp), tp/(fn+tp)
    print("TP : ", tp)
    print("FP : ", fp)
    print("TN : ", tn)
    print("FN : ", fn)

    # Write to CSV file
    with open('./Result_Baseline_Hyperparameters.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([baseline, black_model, dataset , max_level, n_beam, rho, eta, round(tp, 4), round(fp, 4), round(tn, 4), round(fn, 4), round(completeness, 4), round(correctness, 4), round(fidelity, 4), round(robustness, 4), round(accuracy, 4), round(precision, 4), round(recall, 4),round(f1, 4)])
    subset = 'subset'
    print(" =========== The results of ( {baseline}_{dataset}_{subset}_{black_model} ) have been written to 'Result_baseline.csv' ('./baseline/) =========== ".format(baseline=baseline, dataset=dataset, subset=subset, black_model=black_model))