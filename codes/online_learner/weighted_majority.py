import numpy as np

from codes.online_learner.abstract_online_learner import AbstractOnlineLearner


class WeightedMajority(AbstractOnlineLearner):
    def __init__(self, experts, eta):
        super(WeightedMajority, self).__init__(experts)
        self.eta = eta
        self.wt = np.ones(len(experts)) / len(experts)
        self.expert_suggestions = []

    def respond(self, x):
        self.expert_suggestions = [expert.suggest(x) for expert in self.experts]

        choice = np.random.choice(list(range(self.wt.shape[0])), p=self.wt)
        return self.expert_suggestions[choice]

    def learn(self, y):
        zt = y != np.array(self.expert_suggestions)

        wt_tilde = self.wt * np.exp(-self.eta * zt)
        self.wt = wt_tilde / np.sum(wt_tilde)
