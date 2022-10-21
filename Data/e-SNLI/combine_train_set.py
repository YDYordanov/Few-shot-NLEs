"""
Combine esnli_train_1.csv and esnli_train_2.csv into esnli_train.csv
"""

import csv
from tqdm import tqdm


def open_data(data_path):
    with open(data_path, 'r') as e_snli_file:
        reader = csv.DictReader(e_snli_file)
        data = list(reader)
    return data


data1 = open_data('esnli_train_1.csv')
data2 = open_data('esnli_train_2.csv')
data = data1 + data2

print(list(data[0].keys()))

with open('esnli_train.csv', 'w') as write_file:
    writer = csv.DictWriter(write_file, fieldnames=list(data[0].keys()))
    print('writing...')
    writer.writeheader()
    for dat in tqdm(data):
        writer.writerow(dat)
