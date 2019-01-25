import argparse

from codes.cmd.split import split_original_dataset_to_online_offline
from codes.cmd.train_experts import train_experts
from codes.cmd.go_online import go_online


if __name__ == '__main__':
    jump_table = {
        'split': split_original_dataset_to_online_offline,
        'train_experts': train_experts,
        'go_online': go_online,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, type=str, choices=jump_table.keys())
    parser.add_argument('--online_size', required=False, type=float, default=0.95)
    args = parser.parse_args()

    jump_table[args.mode](args)
