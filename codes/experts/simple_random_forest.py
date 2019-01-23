from sklearn.ensemble import RandomForestClassifier

from codes.experts.abstract_expert import AbstracExpert


class SimpleRandomForest(AbstracExpert):
    def __init__(self):
        super(SimpleRandomForest, self).__init__()
        self.model = RandomForestClassifier()

    def train(self, X, G):
        self.model.fit(X, G)

    def suggest(self, x):
        x = x.reshape(1, -1)
        g_hat = self.model.predict(x)
        return g_hat[0]
