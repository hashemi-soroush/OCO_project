from codes.data_utils.read_write_split import read_offline_dataset, read_online_dataset
from codes.experts.simple_linear_regression import SimpleLinearRegressor


def train_experts():
    X, G = read_offline_dataset()

    simple_linear_regressor = SimpleLinearRegressor()
    simple_linear_regressor.train(X, G)

    X2, G2 = read_online_dataset()
    print(simple_linear_regressor.calculate_offline_error(X2, G2))
