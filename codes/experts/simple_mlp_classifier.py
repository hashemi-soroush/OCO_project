from sklearn.neural_network import MLPClassifier

from codes.experts.abstract_expert import AbstracExpert


class SimpleMLPClassifier(AbstracExpert):
    def __init__(self):
        super(SimpleMLPClassifier, self).__init__('simple_MLP_Classifier')
        self.model = MLPClassifier(hidden_layer_sizes=(100, 80, 50, len(self.LABELS)))

    def train(self, X, G):
        self.model.fit(X, G)

    def suggest(self, x):
        x = x.reshape(1, -1)
        g_hat = self.model.predict(x)
        return g_hat[0]
