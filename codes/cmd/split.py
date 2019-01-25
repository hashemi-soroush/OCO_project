from codes.data_utils.read_write_original import read_original_datasets
from codes.data_utils.online_offline_split import split_online_offline, save_split_dataset


def split_original_dataset_to_online_offline(args):
    print('splitting original dataset to online/offline datasets ... ', end='')

    original_dataset = read_original_datasets()

    online_dataset, offline_dataset = split_online_offline(original_dataset, args.offline_size)

    save_split_dataset(online_dataset, 'online')
    save_split_dataset(offline_dataset, 'offline')

    print('OK')
