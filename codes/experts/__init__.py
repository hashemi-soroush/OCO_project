from codes.experts.simple_linear_regression import SimpleLinearRegressor
from codes.experts.simple_mlp_classifier import SimpleMLPClassifier
from codes.experts.simple_random_forest import SimpleRandomForest
from codes.experts.simple_svm import SimpleSVM


def get_all_experts():
    return [
        # SimpleLinearRegressor(),
        SimpleRandomForest(),
        SimpleSVM(),
        SimpleMLPClassifier(),
    ]
