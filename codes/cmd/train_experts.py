from datetime import datetime

from codes.data_utils.read_write_split import read_offline_dataset, read_online_dataset
from codes.experts.simple_linear_regression import SimpleLinearRegressor
from codes.experts.simple_random_forest import SimpleRandomForest
from codes.experts.simple_svm import SimpleSVM


def train_experts():
    X, G = read_offline_dataset()

    print('training simple linear regressor \t {0}'.format(datetime.now()))
    simple_linear_regressor = SimpleLinearRegressor()
    simple_linear_regressor.train(X, G)
    simple_linear_regressor.save_model()

    # print('testing simple linear regressor \t {0}'.format(datetime.now()))
    # X2, G2 = read_online_dataset()
    # print(simple_linear_regressor.calculate_offline_loss(X2, G2)[-1])

    print('training simple random forest \t {0}'.format(datetime.now()))
    simple_random_forest = SimpleRandomForest()
    simple_random_forest.train(X, G)
    simple_random_forest.save_model()

    # print('testing simple random forest \t {0}'.format(datetime.now()))
    # X2, G2 = read_online_dataset()
    # print(simple_random_forest.calculate_offline_loss(X2, G2)[-1])

    print('training simple svm \t {0}'.format(datetime.now()))
    simple_svm = SimpleSVM()
    simple_svm.train(X, G)
    simple_svm.save_model()

    # print('testing simple svm \t {0}'.format(datetime.now()))
    # X2, G2 = read_online_dataset()
    # print(simple_svm.calculate_offline_loss(X2, G2)[-1])
