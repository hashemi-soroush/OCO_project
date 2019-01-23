import numpy as np


class AbstracExpert:
    LABELS = {'a', 'c', 'd', 'e', 'f', 'g', 'h', 'l', 'p', 'r'}

    def __init__(self):
        self.label2id = {}
        self.id2label = {}
        for i, label in enumerate(self.LABELS):
            self.label2id[label] = i
            self.id2label[i] = label

    def labels_2_one_hot_vectors(self, labels):
        one_hot_vectors = np.zeros((len(labels), len(self.LABELS)), np.float32)
        label_ids = [self.label2id[label] for label in labels]
        one_hot_vectors[list(range(len(label_ids))), label_ids] = 1.0
        return one_hot_vectors

    def one_hot_vectors_2_labels(self, one_hot_vectors):
        inds = np.argmax(one_hot_vectors, axis=1)
        labels = [self.id2label[ind] for ind in inds]
        return labels

    def train(self, X, G):
        raise NotImplementedError

    def suggest(self, x):
        raise NotImplementedError

    def calculate_offline_error(self, X, G):
        Y_hat = [self.suggest(x) for x in X]
        G_hat = self.one_hot_vectors_2_labels(Y_hat)

        G = np.array(G)
        error_count = np.sum(G != G_hat)

        error_rate = error_count * 1.0 / len(G_hat)

        return error_rate
