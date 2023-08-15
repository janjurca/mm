import argparse
import json
import os
from sklearn.model_selection import train_test_split
import shutil
import random
parser = argparse.ArgumentParser(description='Split dataset into train, validation and test sets')
parser.add_argument('--dataset', type=str, help='Path to dataset folder')
args = parser.parse_args()


def load_data(manifest):
    random.seed(42)
    data = []
    with open(manifest) as f:
        for line in f:
            data.append(json.loads(line))
    media_ids = [x["media_id"] for x in data]

    counts_per_media = {}
    for media_id in media_ids:
        counts_per_media[media_id] = counts_per_media.get(media_id, 0) + 1

    total_counts = sum(counts_per_media.values())

    sorted_list_descending = sorted(counts_per_media.items(), key=lambda item: item[1], reverse=True)
    print(sorted_list_descending)

    train_media_ids = []
    train_counts = 0
    valid_media_ids = []
    valid_counts = 0
    test_media_ids = []
    test_counts = 0
    for media_id, count in sorted_list_descending:
        if train_counts < 0.8 * total_counts:
            train_media_ids.append(media_id)
            train_counts += count
        elif valid_counts < 0.1 * total_counts:
            valid_media_ids.append(media_id)
            valid_counts += count
        else:
            test_media_ids.append(media_id)
            test_counts += count
    train_data = [x for x in data if x["media_id"] in train_media_ids]
    valid_data = [x for x in data if x["media_id"] in valid_media_ids]
    test_data = [x for x in data if x["media_id"] in test_media_ids]

    return train_data, valid_data, test_data


train_data, valid_data, test_data = load_data(args.dataset + '/all_endoscopy.jsonl')

os.makedirs(args.dataset + '/train', exist_ok=True)
with open(args.dataset + '/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')
        shutil.copy(args.dataset + '/all_endoscopy/' + item['media_id'] + '_' + str(item['frame']) + '.jpg', args.dataset + '/train/' + item['media_id'] + '_' + str(item['frame']) + '.jpg')


os.makedirs(args.dataset + '/valid', exist_ok=True)
with open(args.dataset + '/valid.jsonl', 'w') as f:
    for item in valid_data:
        f.write(json.dumps(item) + '\n')
        shutil.copy(args.dataset + '/all_endoscopy/' + item['media_id'] + '_' + str(item['frame']) + '.jpg', args.dataset + '/valid/' + item['media_id'] + '_' + str(item['frame']) + '.jpg')

os.makedirs(args.dataset + '/test', exist_ok=True)
with open(args.dataset + '/test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
        shutil.copy(args.dataset + '/all_endoscopy/' + item['media_id'] + '_' + str(item['frame']) + '.jpg', args.dataset + '/test/' + item['media_id'] + '_' + str(item['frame']) + '.jpg')
