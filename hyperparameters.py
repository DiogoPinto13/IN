hyperparameters = {    
    "random_forest": [
        {"n_estimators": 5, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 7, "max_depth": 10, "min_samples_split": 4},
        {"n_estimators": 10, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 13, "max_depth": 20, "min_samples_split": 5},
        {"n_estimators": 15, "max_depth": 15, "min_samples_split": 3}
    ],

    "decision_tree": [
        {"max_depth": 3, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 4},
        {"max_depth": 10, "min_samples_split": 2},
        {"max_depth": None, "min_samples_split": 2},
        {"max_depth": 8, "min_samples_split": 5}
    ],

    "linear_regression": [
        {"fit_intercept": True, "positive": False},  # Default
        {"fit_intercept": False, "positive": False},
        {"fit_intercept": True, "positive": True},
        {"fit_intercept": False, "positive": True},
    ],

    "adaboost": [
        {"n_estimators": 5, "learning_rate": 1.0},
        {"n_estimators": 7, "learning_rate": 0.5},
        {"n_estimators": 10, "learning_rate": 0.1},
        {"n_estimators": 13, "learning_rate": 1.5},
        {"n_estimators": 15, "learning_rate": 0.3}
    ],

    "xgboost": [
        {"n_estimators": 5, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 7, "learning_rate": 0.05, "max_depth": 5},
        {"n_estimators": 10, "learning_rate": 0.2, "max_depth": 4},
        {"n_estimators": 13, "learning_rate": 0.01, "max_depth": 6},
        {"n_estimators": 15, "learning_rate": 0.3, "max_depth": 2}
    ],

    "random_forest_regressor": [
        {"n_estimators": 5, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 7, "max_depth": 10, "min_samples_split": 4},
        {"n_estimators": 10, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 13, "max_depth": 5, "min_samples_split": 5},
        {"n_estimators": 15, "max_depth": None, "min_samples_split": 3}
    ],

    "ensemble_trees": [
        {"n_estimators": 5 },
        {"n_estimators": 7 },
        {"n_estimators": 10 },
        {"n_estimators": 13 },
        {"n_estimators": 15 }
    ],
}
