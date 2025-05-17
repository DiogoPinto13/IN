import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, mean_absolute_error
from hyperparameters import hyperparameters
from pre_process_dataset import pre_process_dataset
from sklearn.model_selection import train_test_split

# --- Your model functions ---
from models import adaboost, decision_tree, kmeans, knn, linear_regression, random_forest, xgboost, random_forest_regressor

question_1 = {
  "description": "Days for reporting a crime",
  "models_list": [linear_regression, random_forest_regressor], # random_forest
  "label": "days_difference"
}
question_2 = {
  "description": "Predict the crime severity",
  "models_list": [kmeans],
  "label": "severity_code"
}
question_3 = {
  "description": "Predict the status of a crime",
  "models_list": [random_forest, decision_tree, adaboost, xgboost],
  "label": "status_code"
}
questions_list = [question_1, question_2, question_3] 


# --- Settings ---
n_runs = 5
results = {}

for i, question in enumerate(questions_list):
    print(f"Question {i}: " + question["description"])
    if i == 1: continue

    df, labels = pre_process_dataset("crime_dataset.csv", i)
    for model_fn in question["models_list"]:
        print(f"Running model: {model_fn.__name__}")

        results[model_fn.__name__] = []
        if i == 0:
            X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=42, stratify=labels)
        parameters = hyperparameters[model_fn.__name__]
        scores = []
        for parameter in parameters:
            print(f"Running with parameters: {parameter}")

            args = {
                "X_train": X_train,
                "y_train": y_train,
                **parameter
            }
            model = model_fn(args)
            y_pred = model.predict(X_test)
            
            #regression
            if i == 0:
                y_pred = [round(y) for y in y_pred]
                score = mean_absolute_error(y_test, y_pred)
            else:
                score = f1_score(y_test, y_pred, average='macro')
            scores.append(score)

        results[model_fn.__name__] = scores

# --- Plotting ---
os.makedirs("hyperparams_plots", exist_ok=True)

for model_name, model_scores in results.items():
    plt.figure(figsize=(10, 5))

    sns.barplot(x=[f"comb{i+1}" for i in range(len(model_scores))],
                y=model_scores,
                palette="Blues")

    plt.xticks(rotation=45)
    plt.ylabel("F1 Score" if "regressor" not in model_name and "regression" not in model_name else "MAE")
    plt.title(f"Hyperparameter Combinations - {model_name}")
    plt.tight_layout()
    plt.savefig(f"hyperparams_plots/{model_name}.png")
    plt.show()
