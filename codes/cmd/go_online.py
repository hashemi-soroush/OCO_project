from datetime import datetime

from codes.experts import get_all_experts
from codes.online_learner.weighted_majority import WeightedMajority
from codes.data_utils.read_write_split import read_online_dataset
from codes.enemy.enemy import Enemy


def go_online():
    X, G = read_online_dataset()
    enemy = Enemy(X, G)

    print('loading experts models \t {0}'.format(datetime.now()))
    experts = get_all_experts()
    for expert in experts:
        expert.load_model()

    print('starting online learner \t {0}'.format(datetime.now()))
    online_learner = WeightedMajority(experts, 0.1)
    total_loss = online_learner.start_game(enemy)

    print(total_loss / X.shape[0])
