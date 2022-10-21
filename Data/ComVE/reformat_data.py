import csv
import random
import collections


def process_data(data_path, random_seed=9323280, remove_caps_examples=False,
                 drop_nles=False):
    with open(data_path, 'r') as fi:
        reader = csv.DictReader(fi, delimiter=',')
        all_data = list(reader)

    # Select only the relevant columns
    # Shuffle (seeded) the (training) data
    # Note: drop examples that have all-caps
    random.seed(random_seed)
    print('Data size:', len(all_data))
    id_sample = random.choices([0, 1], k=len(all_data))
    new_data = []
    final_selected_ids = []
    for current_id, (dat, sampled_id) in enumerate(zip(all_data, id_sample)):
        new_dat = dict(dat)
        new_dat.pop('Confusing Reason1', None)
        new_dat.pop('Confusing Reason2', None)
        options = [
            new_dat['Correct Statement'],
            new_dat['Incorrect Statement']
        ]
        shuffled_options = [options[sampled_id], options[1-sampled_id]]
        new_dat.pop('Correct Statement', None)
        new_dat.pop('Incorrect Statement', None)
        new_dat['Sentence1'] = shuffled_options[0]
        new_dat['Sentence2'] = shuffled_options[1]
        new_dat['Correct option'] = sampled_id

        if drop_nles:
            new_dat['Right Reason1'] = ''
            new_dat['Right Reason2'] = ''
            new_dat['Right Reason3'] = ''

        # Remove ALL-CAPS text (corresponding to bad spelling)
        all_caps_found = False
        for key in new_dat.keys():
            text = new_dat[key]
            if isinstance(text, str):
                if text.upper() == text:
                    all_caps_found = True
                new_dat[key] = new_dat[key].capitalize()
        if all_caps_found and remove_caps_examples:
            # remove this all-caps example if remove_caps_examples
            continue

        new_dat = collections.OrderedDict(new_dat)
        new_data.append(new_dat)
        final_selected_ids.append(current_id)

    print('New data size:', len(new_data))
    return new_data, final_selected_ids


def write_data(data_list_ordered_dict, data_path):
    with open(data_path, 'w') as f:
        writer = csv.DictWriter(
            f, delimiter=',', fieldnames=data_list_ordered_dict[0].keys())
        writer.writeheader()
        writer.writerows(data_list_ordered_dict)


if __name__ == "__main__":
    train_data_no_nles, _ = process_data(
        'train.csv', random_seed=4904095, drop_nles=True)
    write_data(train_data_no_nles, data_path='train_no_nles.csv')

    train_data, _ = process_data('train.csv', random_seed=45923)

    clean_data, selected_ids = process_data(
        'train.csv', random_seed=215050, remove_caps_examples=True)
    random.seed(4539834059)
    train_ids_1000 = random.sample(selected_ids, 1000)
    train_nles_1000 = [dat for idx, dat in enumerate(train_data)
                       if idx in train_ids_1000]
    write_data(train_nles_1000, data_path='train_1000_nles_only.csv')

    train_ids_50 = train_ids_1000[:50]
    train_nles_50 = [dat for idx, dat in enumerate(train_data)
                     if idx in train_ids_50]
    write_data(train_nles_50, data_path='train_50_nles_only.csv')
    
    train_ids_25 = train_ids_1000[:25]
    train_nles_25 = [dat for idx, dat in enumerate(train_data)
                     if idx in train_ids_25]
    write_data(train_nles_25, data_path='train_25_nles_only.csv')
    
    train_ids_100 = train_ids_1000[:100]
    train_nles_100 = [dat for idx, dat in enumerate(train_data)
                     if idx in train_ids_100]
    write_data(train_nles_100, data_path='train_100_nles_only.csv')
    
    train_ids_200 = train_ids_1000[:200]
    train_nles_200 = [dat for idx, dat in enumerate(train_data)
                     if idx in train_ids_200]
    write_data(train_nles_200, data_path='train_200_nles_only.csv')

    train_50_nles_data = [dat if idx not in train_ids_50 else train_data[idx]
                          for idx, dat in enumerate(train_data_no_nles)]
    write_data(train_50_nles_data, data_path='train_50_nles.csv')

    data, _ = process_data('dev.csv', random_seed=4354341)
    write_data(data, data_path='dev.csv')

    test_data, _ = process_data('test.csv', random_seed=98750435)
    write_data(test_data, data_path='test.csv')
    
    random.seed(5367894)
    test_ids_100 = random.sample(list(range(len(test_data))), 100)
    test_nles_100 = [dat for idx, dat in enumerate(test_data)
                       if idx in test_ids_100]
    write_data(test_nles_100, data_path='test_100.csv')

