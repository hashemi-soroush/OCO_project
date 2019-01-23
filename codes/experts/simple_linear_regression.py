from sklearn.linear_model import LinearRegression

from codes.experts.abstract_expert import AbstracExpert


class SimpleLinearRegressor(AbstracExpert):
    def __init__(self):
        super(SimpleLinearRegressor, self).__init__('simple_linear_regressor')
        self.model = LinearRegression()

    def train(self, X, G):
        Y = self.labels_2_one_hot_vectors(G)
        self.model.fit(X, Y)

    def suggest(self, x):
        x = x.reshape(1, -1)
        y_hat = self.model.predict(x)
        g_hat = self.one_hot_vector_2_label(y_hat[0])
        return g_hat
