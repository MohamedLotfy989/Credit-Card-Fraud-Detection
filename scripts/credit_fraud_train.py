import argparse
import pickle
import os
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid, RandomizedSearchCV
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from credit_fraud_utils_data import load_data
from credit_fraud_utils_eval import evaluate_model,load_best_params
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def train_logistic_regression(X_train_scaled, y_train, random_seed=42):
    param_grid = {
        'C': [0.1 ,1.0 ,10.0],  # Keep this range
        'penalty': ['l2'],
        'class_weight': [{0: 0.2, 1: 0.8},{0: 0.1, 1: 0.9},{0: 0.33, 1: 0.67}],
        'solver': ['sag','saga', 'lbfgs'],
        'max_iter': [300,500,800],
    }

    lr = LogisticRegression()
    scorer = make_scorer(f1_score, pos_label=1)

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(lr, param_grid, cv=stratified_kfold, scoring=scorer, n_jobs=-1)

    # Calculate total iterations:
    # Number of parameter combinations * number of CV folds.
    n_candidates = len(list(ParameterGrid(param_grid)))
    total_iterations = n_candidates * stratified_kfold.get_n_splits(y_train)

    # Use tqdm_joblib to wrap the grid search fitting process with a progress bar.
    with tqdm_joblib(tqdm(total=total_iterations, desc="GridSearchCV")):
        grid_search.fit(X_train_scaled, y_train)

    # grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    # best_params =   {'C': 10.0, 'class_weight': {0: 0.2, 1: 0.8}, 'max_iter': 500, 'penalty': 'l2', 'solver': 'sag'}

    lr = LogisticRegression(**best_params, random_state=random_seed)
    lr.fit(X_train_scaled, y_train)

    return lr, best_params

def train_random_forest(X_train, y_train, random_seed=42):


    # Define a parameter distribution for Random Forest
    param_distributions = {
        'n_estimators': [ 100, 200,400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': [None, {0: 0.2, 1: 0.8}],
    }

    # Initialize the base RandomForestClassifier
    rf = RandomForestClassifier(random_state=random_seed)

    # Use f1 score (with pos_label=1) as the scoring metric
    scorer = make_scorer(f1_score, pos_label=1)

    # Setup StratifiedKFold for cross-validation
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Number of random parameter settings to try
    n_iter_search = 50
    total_iterations = n_iter_search * stratified_kfold.get_n_splits(y_train)

    # Setup RandomizedSearchCV with the defined parameter distributions
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        cv=stratified_kfold,
        scoring=scorer,
        random_state=random_seed,
        n_jobs=-1,
        verbose=0
    )

    # Wrap the fitting process with a progress bar using tqdm_joblib
    with tqdm_joblib(tqdm(total=total_iterations, desc="RandomizedSearchCV")):
        random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print("Best Hyperparameters for Random Forest:", best_params)


    # best_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None, 'class_weight': {0: 0.2, 1: 0.8}, 'bootstrap': False}

    # Create a new RandomForestClassifier with the best found parameters and train it on the full training data
    best_rf = RandomForestClassifier(**best_params, random_state=random_seed)
    best_rf.fit(X_train, y_train)

    return best_rf, best_params

def train_xgboost(X_train, y_train, random_seed=42):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 2, 3]
    }

    xgb = XGBClassifier(random_state=random_seed)
    scorer = make_scorer(f1_score, pos_label=1)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=stratified_kfold,
        scoring=scorer,
        n_jobs=-1,
        verbose=0
    )

    total_iterations = len(list(ParameterGrid(param_grid))) * stratified_kfold.get_n_splits(y_train)

    with tqdm_joblib(tqdm(total=total_iterations, desc="GridSearchCV for XGBoost")):
        grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters for XGBoost:", best_params)

    # best_params = {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 200, 'scale_pos_weight': 2, 'subsample': 0.6}

    best_xgb = XGBClassifier(**best_params, random_state=random_seed)
    best_xgb.fit(X_train, y_train)

    return best_xgb, best_params

def train_voting_classifier(X_train, y_train, lr_params, rf_params):
    # Initialize the LogisticRegression and RandomForestClassifier with the provided parameters
    lr = LogisticRegression(**lr_params, random_state=42)
    rf = RandomForestClassifier(**rf_params, random_state=42)

    # Define the VotingClassifier with 'soft' voting.
    voting_clf = VotingClassifier(estimators=[("lr", lr), ("rf", rf)], voting="soft")

    # Define the parameter grid for tuning only the ensemble weights.
    param_grid = {
        "weights": [[0.25, 0.75], [0.1, 0.9], [0.33, 0.67]]
    }

    # Use f1 score (with pos_label=1) as the scoring metric.
    scorer = make_scorer(f1_score, pos_label=1)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Total iterations for progress bar (number of weight combinations * number of CV folds)
    total_iterations = len(param_grid["weights"]) * stratified_kfold.get_n_splits(y_train)

    # Set up GridSearchCV to search over the weights.
    grid_search = GridSearchCV(
        estimator=voting_clf,
        param_grid=param_grid,
        cv=stratified_kfold,
        scoring=scorer,
        n_jobs=-1,
        verbose=0
    )

    # Wrap the fitting process with a progress bar.
    with tqdm_joblib(tqdm(total=total_iterations, desc="VotingClassifier Search")):
        grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best ensemble weights:", best_params)

    # best_params =   {'weights': [0.25, 0.75]}

    # Build and train a new VotingClassifier using the tuned weights.
    best_voting = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=best_params["weights"]
    )
    best_voting.fit(X_train, y_train)

    return best_voting, best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for credit card fraud detection")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--model", type=str, choices=["logistic", "random_forest", "xgboost", "voting"],default="logistic", required=True, help="Model type")
    parser.add_argument("--output_path", type=str, default="./output/model.pkl", required=True, help="Path to save the trained model")
    parser.add_argument("--config_path", type=str, default="./configs/best_model_params.yml", help="Path to the YAML config file")
    args = parser.parse_args()
    # Setup Logger
    # Load Data
    X_train, y_train, X_val, y_val = load_data(args.train_data_path, args.val_data_path)

    # Handle Imbalanced Data with SMOTE
    # X_train_resampled, y_train_resampled = resample_data(X_train, y_train, technique="smote") # SMOTE

    # Load best parameters from YAML file
    best_params = load_best_params(args.config_path)

    # Model Selection
    if args.model == "logistic":
        model, best_params = train_logistic_regression(X_train, y_train)
    elif args.model == "random_forest":
        model, best_params = train_random_forest(X_train, y_train)
    elif args.model == "xgboost":
        model, best_params = train_xgboost(X_train, y_train)
    elif args.model == "voting":
        lr_params = best_params['logistic_regression']
        rf_params = best_params['random_forest']
        voting_params = best_params['voting']
        model, best_params = train_voting_classifier(X_train, y_train, lr_params, rf_params)

    # Evaluate Model
    metrics = evaluate_model(model, X_val, y_val, args.output_path, args.model)

    print(f"Validation Metrics: {metrics}, Best Params: {best_params}")
    # Extract best_threshold from metrics
    best_threshold = metrics["optimized"]["best_threshold"]

    # Save Model
    model_file_path = os.path.join(args.output_path, "model.pkl")
    with open(model_file_path, "wb") as f:
        pickle.dump({"model": model, "threshold": best_threshold}, f)
    print(f"Model and threshold saved at {args.output_path}")



