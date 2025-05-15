import pandas as pd
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

def decision_tree(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def linear_regression(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def random_forest(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    n_estimators = args["n_estimators"]
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def knn(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    n_neighbors = args["n_neighbors"]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def validation(model, X_test, y_test, question_regression):
    predictions = model.predict(X_test)
    if question_regression == "days_difference":
        return {
            "mean absolute error": mean_absolute_error(y_test, predictions),
            "mean squarred error": mean_squared_error(y_test, predictions)
        }
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, labels=y_test, average="macro"),
        "recall": recall_score(y_test, predictions, labels=y_test, average="macro"),
        "f1_score": f1_score(y_test, predictions, labels=y_test, average="macro"),
        "model": model.__name__
    }

def delete_features(df, features_list):
    df = df.copy()
    feautres = [feature for feature in features_list if feature in df.columns]
    df.drop(columns=feautres, inplace=True)
    return df

def label_encode_columns(df, columns):
    df_encoded = df.copy()
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def impute_with_value(df, column_list, default_value):
    df_imputed = df.copy()
    for column in column_list:
        df_imputed[column] = df_imputed[column].fillna(default_value)
    return df_imputed

def pre_process_dataset(dataset_path, label):
    df = pd.read_csv(dataset_path, sep=";")
    #print(df.columns.values)
    #remove the useless features
    df = delete_features(df, ["mo_code"])
    df = impute_with_value(df, ["weapon_code", "crime_type_code2"], 0)
    df = label_encode_columns(df, df.columns.values)#["vict_sex", "vict_descent_code"])
    #get the label column in the dataframe according to the question
    labels = df.pop(label)
    #normalize the data
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return (df_normalized, label)

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
        "n_estimators": 13,
        "n_neighbors": 5
        }
    
    question_1 = {
        "description": "Days for reporting a crime",
        "models_list": [linear_regression, random_forest],
        "label": "days_difference"
    }
    question_2 = {
        "description": "Predict the crime severity",
        "models_list": [knn],
        "label": "severity_code"
    }
    question_3 = {
        "description": "Predict the status of a crime",
        "models_list": [knn, random_forest, decision_tree],
        "label": "Status"
    }
    questions_list = [question_1, question_2, question_3]
    for question in questions_list:
        df_normalized, labels = pre_process_dataset("crime_dataset.csv", question["label"])
        print("Question: " + question["description"])
        for model in question["models_list"]:
            for i in range(1):
                X_train, X_test, y_train, y_test = train_test_split(df_normalized, labels, test_size=0.3, random_state=i)
                args["X_train"] = X_train
                args["y_train"] = y_train
                performance_metrics.append(validation(model(args), X_test, y_test, question["label"]))

        save_results(performance_metrics, question)
        performance_metrics.clear()

if __name__ == "__main__":
    #df = pd.read_csv("crime_dataset.csv", sep=";")
    #print(df.head(20))
    main()