# Utility Script: credit_fraud_utils_eval.py
# Added threshold optimization.
import logging
import os

import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
import numpy as np


def evaluate_model(model, X_val, y_val, output_dir, model_type):
    """
    Evaluate the model performance using F1-Score and PR-AUC with both default and optimized thresholds.

    Args:
        model: Trained machine learning model.
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation labels.
        output_dir (str): Directory to save evaluation reports.
        model_type (str): Type of the model used.

    Returns:
        dict: Dictionary containing F1-Score and PR-AUC for both default and optimized thresholds.
    """
    # Setup Logger
    log_file_path = os.path.join(output_dir, f"{model_type}_evaluation.log")
    logger = setup_logger(log_file_path)

    # Default threshold evaluation
    y_pred_default = model.predict(X_val)
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    f1_default = f1_score(y_val, y_pred_default)
    precision_default, recall_default, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_auc_default = auc(recall_default, precision_default)

    # Save PR-AUC plot and confusion matrix for default threshold
    pr_auc_plot_path_default = os.path.join(output_dir, f"{model_type}_pr_auc_plot_default.png")
    save_pr_auc_plot(y_val, y_pred_prob, pr_auc_plot_path_default)

    confusion_matrix_path_default = os.path.join(output_dir, f"{model_type}_confusion_matrix_default.png")
    save_confusion_matrix(y_val, y_pred_default, confusion_matrix_path_default)

    # Optimized threshold evaluation
    best_threshold = optimize_threshold(model, X_val, y_val)
    y_pred_optimized = (y_pred_prob >= best_threshold).astype(int)

    f1_optimized = f1_score(y_val, y_pred_optimized)
    precision_optimized, recall_optimized, _ = precision_recall_curve(y_val, y_pred_prob)
    pr_auc_optimized = auc(recall_optimized, precision_optimized)

    # Save PR-AUC plot and confusion matrix for optimized threshold
    pr_auc_plot_path_optimized = os.path.join(output_dir, f"{model_type}_pr_auc_plot_optimized.png")
    save_pr_auc_plot(y_val, y_pred_prob, pr_auc_plot_path_optimized)

    confusion_matrix_path_optimized = os.path.join(output_dir, f"{model_type}_confusion_matrix_optimized.png")
    save_confusion_matrix(y_val, y_pred_optimized, confusion_matrix_path_optimized)

    # Generate and save classification report for both thresholds
    report_default = classification_report(y_val, y_pred_default, output_dict=True)
    report_optimized = classification_report(y_val, y_pred_optimized, output_dict=True)

    report_path_default = os.path.join(output_dir, f"{model_type}_classification_report_default.txt")
    report_path_optimized = os.path.join(output_dir, f"{model_type}_classification_report_optimized.txt")

    with open(report_path_default, "w") as f:
        f.write(classification_report(y_val, y_pred_default))

    with open(report_path_optimized, "w") as f:
        f.write(classification_report(y_val, y_pred_optimized))

    # Log the best threshold, metrics, and model type
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Best Threshold: {best_threshold}")
    logger.info(f"Default Threshold Metrics: F1-Score = {f1_default}, PR-AUC = {pr_auc_default}")
    logger.info(f"Optimized Threshold Metrics: F1-Score = {f1_optimized}, PR-AUC = {pr_auc_optimized}")

    # Log macro avg and precision/recall for positive class
    logger.info(f"Default Threshold Macro Avg: {report_default['macro avg']}")
    logger.info(f"Default Threshold Precision for Positive Class: {report_default['1']['precision']}")
    logger.info(f"Default Threshold Recall for Positive Class: {report_default['1']['recall']}")
    logger.info(f"Optimized Threshold Macro Avg: {report_optimized['macro avg']}")
    logger.info(f"Optimized Threshold Precision for Positive Class: {report_optimized['1']['precision']}")
    logger.info(f"Optimized Threshold Recall for Positive Class: {report_optimized['1']['recall']}")

    return {
        "default": {"f1_score": f1_default, "pr_auc": pr_auc_default},
        "optimized": {"f1_score": f1_optimized, "pr_auc": pr_auc_optimized, "best_threshold": best_threshold}
    }



def save_pr_auc_plot(y_true, y_pred_prob, output_path):
    """
     Save the Precision-Recall AUC plot and Precision-Recall vs Threshold plot.

     Args:
         y_true (pd.Series): True labels.
         y_pred_prob (np.ndarray): Predicted probabilities.
         output_path (str): Path to save the PR-AUC plot.
     """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall AUC')
    plt.legend(loc='best')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Save Precision-Recall vs Threshold plot
    plt.figure()
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend(loc='best')
    pr_threshold_plot_path = output_path.replace(".png", "_thresholds.png")
    plt.savefig(pr_threshold_plot_path)
    plt.close()


def optimize_threshold(model, X_val, y_val):
    """
    Optimize decision threshold for best F1-Score.

    Args:
        model: Trained machine learning model.
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation labels.

    Returns:
        float: Best threshold for F1-Score.
    """
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.0, 1.1, 0.01)
    best_threshold, best_f1 = 0, 0

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold




def save_confusion_matrix(y_true, y_pred, output_path):
    """
    Save the confusion matrix plot.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        output_path (str): Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)


    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    sns.heatmap(cm_percentage, annot=True, fmt='.1f', ax=ax2, cmap='Blues')
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_ylabel('True Predicted')
    ax2.set_xlabel('Actual Label')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def setup_logger(log_file_path):
    """
    Setup logger for the project.

    Args:
        log_file_path (str): Path to save the log file.
    """
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger


def load_best_params(yaml_path):
    with open(yaml_path, 'r') as file:
        best_params = yaml.safe_load(file)

    # Convert class_weight keys from strings to integers
    if 'logistic_regression' in best_params and 'class_weight' in best_params['logistic_regression']:
        best_params['logistic_regression']['class_weight'] = {
            int(k): v for k, v in best_params['logistic_regression']['class_weight'].items()
        }
    if 'random_forest' in best_params and 'class_weight' in best_params['random_forest']:
        best_params['random_forest']['class_weight'] = {
            int(k): v for k, v in best_params['random_forest']['class_weight'].items()
        }

    return best_params

