from codes.experts.simple_linear_regression import SimpleLinearRegressor
from codes.experts.simple_random_forest import SimpleRandomForest


def get_all_experts():
    return [
        SimpleLinearRegressor(),
        SimpleRandomForest()
    ]
