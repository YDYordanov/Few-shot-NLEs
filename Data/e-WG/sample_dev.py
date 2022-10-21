"""
Take a random sample from train_xl to be the dev set. Also create a new train file.
The split preserves pairs.
"""
import random
import jsonlines


train_path = 'train_xl.jsonl'
with jsonlines.open(train_path) as reader:
    train_data = list(reader)

dev_size = 1268
random.seed(2089)
data_size = len(train_data)
# Preserve the pairs when doing split
print('Num pairs:', len(train_data) // 2)
id_list = list(range(len(train_data) // 2))
dev_ids = random.sample(id_list, dev_size)
dev_ids.sort()
set_diff = set(id_list).difference(set(dev_ids))
new_train_ids = list(set_diff)
new_train_ids.sort()

dev_data = []
for i in dev_ids:
    dev_data += [train_data[2*i], train_data[2*i + 1]]
new_train_data = []
for i in new_train_ids:
    new_train_data += [train_data[2*i], train_data[2*i + 1]]

# Write the new data to file
out_file = 'new_dev.jsonl'
with jsonlines.open(out_file, 'w') as writer:
    writer.write_all(dev_data)

out_file = 'new_train.jsonl'
with jsonlines.open(out_file, 'w') as writer:
    writer.write_all(new_train_data)

