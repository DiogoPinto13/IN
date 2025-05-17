import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from custom_models import KMeansClassifier, TreesEnsemble
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

def random_forest_regressor(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def adaboost(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def xgboost(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

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

def kmeans(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    seed = args["seed"]
    model = KMeansClassifier(seed)
    model.fit(X_train, y_train)
    return model

def ensemble_trees(args):
    X_train = args["X_train"]
    y_train = args["y_train"]
    n_estimators = args["n_estimators"]
    model = TreesEnsemble(n_estimators)
    model.fit(X_train, y_train)
    return model