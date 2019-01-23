import numpy as np


class AbstracExpert:
    LABELS = {'a', 'c', 'd', 'e', 'f', 'g', 'h', 'l', 'p', 'r'}

    def __init__(self):
        self.label2ind = {}
        self.ind2label = {}
        for i, label in enumerate(self.LABELS):
            self.label2ind[label] = i
            self.ind2label[i] = label

    def labels_2_one_hot_vectors(self, labels):
        one_hot_vectors = np.zeros((len(labels), len(self.LABELS)), np.float32)
        label_inds = [self.label2ind[label] for label in labels]
        one_hot_vectors[list(range(len(label_inds))), label_inds] = 1.0
        return one_hot_vectors

    def one_hot_vector_2_label(self, one_hot_vector):
        ind = np.argmax(one_hot_vector)
        label = self.ind2label[ind]
        return label

    def train(self, X, G):
        raise NotImplementedError

    def suggest(self, x):
        raise NotImplementedError

    def calculate_offline_error(self, X, G):
        G_hat = [self.suggest(x) for x in X]

        G = np.array(G)
        error_count = np.sum(G != G_hat)

        error_rate = error_count * 1.0 / len(G_hat)

        return error_rate
