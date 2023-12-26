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

# Evaluate_and_save_results
def evaluate_and_save_results(test_data, test_target, predictions, original_predictions, perturbed_predictions, dataset, subset, baseline='None', black_model=None):
    # Calculate classification metrics
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions, average='binary')
    recall = recall_score(test_target, predictions, average='binary')
    f1 = f1_score(test_target, predictions, average='binary')

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
    with open('/home/lry/pythondata/venv/lry/baseline/Result_baseline_2.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow([baseline, black_model, dataset , subset,tp, fp, tn, fn, completeness, correctness, fidelity, robustness, accuracy, precision, recall, f1])
        writer.writerow([baseline, black_model, dataset , subset, round(tp, 4), round(fp, 4), round(tn, 4), round(fn, 4), round(completeness, 4), round(correctness, 4), round(fidelity, 4), round(robustness, 4), round(accuracy, 4), round(precision, 4), round(recall, 4),round(f1, 4)])
    
    print(" =========== The results of ( {baseline}_{dataset}_{subset}_{black_model} ) have been written to 'Result_baseline.csv' ('/home/lry/pythondata/venv/lry/baseline/) =========== ".format(baseline=baseline, dataset=dataset, subset=subset, black_model=black_model))


# Evaluate_and_save_results
def Evaluate_and_save_results( test_target, predictions, original_predictions, perturbed_predictions, dataset, subset, baseline='None', black_model=None, avg_train_time=0.0, avg_pred_time=0.0):
    # Calculate classification metrics
    accuracy = accuracy_score(test_target, predictions)
    precision = precision_score(test_target, predictions, average='binary') # average='micro'
    recall = recall_score(test_target, predictions, average='binary')
    f1 = f1_score(test_target, predictions, average='binary')

    # Calculate evaluation metrics
    input_instances = len(test_target)

    # Completeness
    covered_by_rules = sum([1 for pred, true_label in zip(predictions, test_target) if pred == true_label])
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
    with open('/home/lry/pythondata/venv/lry/baseline/Result_baseline_2.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow([baseline, black_model, dataset , subset,tp, fp, tn, fn, completeness, correctness, fidelity, robustness, accuracy, precision, recall, f1])
        writer.writerow([baseline, black_model, dataset , subset, round(tp, 4), round(fp, 4), round(tn, 4), round(fn, 4), round(completeness, 4), round(correctness, 4), round(fidelity, 4), round(precision, 4), round(recall, 4), round(robustness, 4), round(accuracy, 4),round(f1, 4),round(avg_train_time, 4),round(avg_pred_time, 10)])
    print(" =========== The results of ( {baseline}_{dataset}_{subset}_{black_model} ) have been written to 'Result_baseline.csv' ('/home/lry/pythondata/venv/lry/baseline/) =========== ".format(baseline=baseline, dataset=dataset, subset=subset, black_model=black_model))

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

# Plot the feature importances of the forest
def plot_feature_importances(importance_values, feature_names, title):
    # plt.rcParams.update({'font.size': 60})
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_values)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot
    plt.show()

# Plot the feature importances of the forest
def delete_feature(X_data, features):
    for feature in features:
        feature_index = CUSTOM_FEAT_COLS.index(feature)
        X_data[:, feature_index] = 0
    return X_data

def permutation_feature_importance_with_rules(pre_model, x, rules):
    """
    Calculate Permutation Feature Importance with rules for the model given a single data point (x),
    rules, and the score computed using OCSVM.
    """

    score = -pre_model.score_samples(x.reshape(1, -1))

    num_features = x.shape[0]
    permutation_importances = np.zeros(num_features)

    for rule in rules:
        feature_id = rule['feature_id']
        feature_value = rule['value']
        feature_threshold = rule['threshold']

        # Calculate the difference between the feature value and the threshold
        feature_diff = (feature_value - feature_threshold) / feature_threshold

        x_permuted = np.copy(x)
        x_permuted[feature_id] = 0 # Deleting this feature
        score_permuted = -pre_model.score_samples(x_permuted.reshape(1, -1))

        # Calculate the importance score using the formula and store it in the array
        permutation_importances[feature_id] = np.abs(np.abs(score) - np.abs(score_permuted)) * np.abs(feature_diff)
    
    permutation_feature_importance_dict = {}
    for i in range(len(CUSTOM_FEAT_COLS)):
        permutation_feature_importance_dict[CUSTOM_FEAT_COLS[i]] = permutation_importances[i]
    sorted_permutation_feature_importance = dict(sorted(permutation_feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    # 使用字典推导式遍历字典并删除值为0的项
    permutation_feature_importance = {k: v for k, v in sorted_permutation_feature_importance.items() if v != 0.0}

    return sorted_permutation_feature_importance
