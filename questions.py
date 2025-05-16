from models import adaboost, decision_tree, knn, linear_regression, random_forest, xgboost, random_forest_regressor

question_1 = {
  "description": "Days for reporting a crime",
  "models_list": [linear_regression, random_forest_regressor], # random_forest
  "label": "days_difference"
}
question_2 = {
  "description": "Predict the crime severity",
  "models_list": [knn],
  "label": "severity_code"
}
question_3 = {
  "description": "Predict the status of a crime",
  "models_list": [random_forest, decision_tree, adaboost, xgboost],
  "label": "status_code"
}
questions_list = [question_1, question_3] 