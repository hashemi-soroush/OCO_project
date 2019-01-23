from sklearn.linear_model import LinearRegression

from codes.experts.abstract_expert import AbstracExpert


class SimpleLinearRegressor(AbstracExpert):
    def __init__(self):
        super(SimpleLinearRegressor, self).__init__()
        self.model = LinearRegression()

    def train(self, X, G):
        Y = self.labels_2_one_hot_vectors(G)
        self.model.fit(X, Y)

    def suggest(self, x):
        x = x.reshape(1, -1)
        y_hat = self.model.predict(x)
        return y_hat[0]
