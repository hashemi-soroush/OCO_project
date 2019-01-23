from datetime import datetime

import matplotlib.pyplot as plt

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

    print('calculating experts loss on online data \t {0}'.format(datetime.now()))
    experts_loss = {}
    for expert in experts:
        print('calculating {0} loss \t {1}'.format(expert.name, datetime.now()))
        experts_loss[expert.name] = expert.calculate_offline_loss(X, G)

    print('starting online learner \t {0}'.format(datetime.now()))
    online_learner = WeightedMajority(experts, 0.1)
    online_learner_loss = online_learner.start_game(enemy)

    compare(experts_loss, online_learner_loss)


def compare(experts_loss, online_learner_loss):
    print('online_learner_loss \t {0}'.format(online_learner_loss[-1]))
    for expert_name, expert_loss in experts_loss.items():
        print('{0} \t {1}'.format(expert_name, expert_loss[-1]))

    draw_cumulative_losses(experts_loss, online_learner_loss)


def draw_cumulative_losses(experts_loss, online_learner_loss):
    for expert_name, expert_loss in experts_loss.items():
        plt.plot(expert_loss, label=expert_name)
    plt.plot(online_learner_loss, label='online_learner')
    plt.legend()
    plt.show()
