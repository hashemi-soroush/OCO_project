from datetime import datetime

from codes.experts import get_all_experts
from codes.data_utils.read_write_split import read_offline_dataset, read_online_dataset


def train_experts(args):
    X, G = read_offline_dataset()
    X2, G2 = read_online_dataset()

    for expert in get_all_experts():
        print('[{0}] \t training {1}'.format(datetime.now(), expert.name))
        expert.train(X, G)
        expert.save_model()

        print('[{0}] \t testing {1}'.format(datetime.now(), expert.name))
        losses = expert.calculate_offline_loss(X2, G2)
        print('[{0}] \t loss of {1} \t {2}'.format(datetime.now(), expert.name, losses[-1]))
