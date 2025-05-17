import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from pre_process_dataset import pre_process_dataset
from questions import questions_list

def validation(model_name, model, X_test, y_test, question_regression):
    predictions = model.predict(X_test)
    if question_regression == "days_difference":
        return {
            "mean absolute error": mean_absolute_error(y_test, predictions),
            "mean squarred error": mean_squared_error(y_test, predictions),
            "model": model_name
        }
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, labels=y_test, average="macro"),
        "recall": recall_score(y_test, predictions, labels=y_test, average="macro"),
        "f1_score": f1_score(y_test, predictions, labels=y_test, average="macro"),
        "model": model_name
    }

def save_results(performance_metrics, question, output_dir="results"):
    question_label = question["label"]
    question_dir = os.path.join(output_dir, question_label)
    os.makedirs(question_dir, exist_ok=True)

    model_results = {}

    for result in performance_metrics:
        model_name = result["model"]
        result_copy = result.copy()
        del result_copy["model"]
        model_results.setdefault(model_name, []).append(result_copy)

    for model_name, results in model_results.items():
        df = pd.DataFrame(results)
        file_path = os.path.join(question_dir, f"{model_name}_results.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved results for {model_name} to {file_path}")

def read_results(results_dir="results"):
    summary_data = []
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.csv"):
            model_name = filename.replace("_results.csv", "")
            file_path = os.path.join(results_dir, filename)

            df = pd.read_csv(file_path)
            metrics_mean = df.mean(numeric_only=True)
            metrics_std = df.std(numeric_only=True)
            
            summary = {
                "model": model_name
            }
            for metric in metrics_mean.index:
                summary[f"{metric}_mean"] = metrics_mean[metric]
                summary[f"{metric}_std"] = metrics_std[metric]

            summary_data.append(summary)

    return pd.DataFrame(summary_data)

def main():
    performance_metrics = list()
    args = {
        'n_clusters':3,
        'fit_intercept': False, 'positive': False,
        'n_estimators_rf_regressor': 7, 'max_depth_rf_regressor': 10, 'min_samples_split_rf_regressor': 4,
        'n_estimators_rf': 15, 'max_depth_rf': 15, 'min_samples_split_rf': 3,
        'max_depth_dt': 10, 'min_samples_split_dt': 2,
        'n_estimators_ada': 5, 'learning_rate_ada': 1.0,
        'n_estimators_xgboost': 10, 'learning_rate_xgboost': 0.2, 'max_depth_xgboost': 4,
        'n_estimators_ensemble': 15
    }
        
    for i, question in enumerate(questions_list):
        df, labels = pre_process_dataset("crime_dataset.csv", i)
        print("Question: " + question["description"])
        for model in question["models_list"]:
            for seed in range(15):
                if i == 0:
                    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=seed)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.3, random_state=seed, stratify=labels)
                args["X_train"] = X_train
                args["y_train"] = y_train
                args["seed"] = seed
                performance_metrics.append(validation(model.__name__, model(args), X_test, y_test, question["label"]))

        save_results(performance_metrics, question)
        performance_metrics.clear()

if __name__ == "__main__":
    df = pd.read_csv("crime_dataset.csv", sep=";")
    print(df.head(20))
    # pre_process_dataset("crime_dataset.csv", 0)
    # pre_process_dataset("crime_dataset.csv", 1)
    # pre_process_dataset("crime_dataset.csv", 2)
    main()