from sklearn.linear_model import SGDClassifier

from codes.experts.abstract_expert import AbstracExpert


class SimpleSVM(AbstracExpert):
    def __init__(self):
        super(SimpleSVM, self).__init__('simple_svm')
        self.model = SGDClassifier()

    def train(self, X, G):
        self.model.fit(X, G)

    def suggest(self, x):
        x = x.reshape(1, -1)
        g_hat = self.model.predict(x)
        return g_hat[0]
