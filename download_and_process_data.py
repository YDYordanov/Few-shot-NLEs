import os
import csv
import zipfile
import shutil
import random
import collections
import jsonlines
import wget

from wget import bar_adaptive
from tqdm import tqdm


def make_dir(dir_path):
    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)


def open_csv_data(data_path):
    with open(data_path, 'r') as e_snli_file:
        reader = csv.DictReader(e_snli_file)
        data = list(reader)
    return data


def process_comve_data(data_path, random_seed=9323280, remove_caps_examples=False,
                 drop_nles=False):
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        all_data = list(reader)

    # Select only the relevant columns
    # Shuffle (seeded) the (training) data
    # Note: drop examples that have all-caps
    random.seed(random_seed)
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

    return new_data, final_selected_ids


def write_comve_data(data_list_ordered_dict, data_path):
    with open(data_path, 'w') as f:
        writer = csv.DictWriter(
            f, delimiter=',', fieldnames=data_list_ordered_dict[0].keys())
        writer.writeheader()
        writer.writerows(data_list_ordered_dict)


def download_esnli_data():
    print('1) Downloading e-SNLI...')
    make_dir('Data/temp/e-SNLI')
    wget.download(
        'https://github.com/OanaMariaCamburu/e-SNLI/archive/refs/heads/master.zip', 
        'Data/temp/e-SNLI', bar='')  # bar_adaptive)
    print('\ne-SNLI downloaded!')
    
    print('\n2) Unpacking e-SNLI data...')
    with zipfile.ZipFile("Data/temp/e-SNLI/e-SNLI-master.zip","r") as zip_ref:
        zip_ref.extractall("Data/temp/e-SNLI")
    print('Data unpacked!')
    
    print('\n3) Moving the data files...')
    data_folder = 'Data/e-SNLI'
    make_dir(data_folder)
    data_files = ['esnli_dev.csv', 'esnli_test.csv', 'esnli_train_1.csv', 'esnli_train_2.csv']
    for file_name in data_files:
        data_file_path = 'Data/temp/e-SNLI/e-SNLI-master/dataset/' + file_name
        shutil.copy(data_file_path, data_folder)
    print('Data moved!')


def download_winogrande_data():
    print('1) Downloading WinoGrande...')
    make_dir('Data/temp/WinoGrande')
    wget.download(
        'https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip', 
        'Data/temp/WinoGrande/winogrande_1.1.zip', bar='')  #bar_adaptive)
    print('\nWinoGrande downloaded!')
    
    print('\n2) Unpacking WinoGrande data...')
    with zipfile.ZipFile("Data/temp/WinoGrande/winogrande_1.1.zip","r") as zip_ref:
        zip_ref.extractall("Data/temp/WinoGrande")
    print('Data unpacked!')
    
    print('\n3) Moving the data files...')
    data_folder = 'Data/e-WG'
    make_dir(data_folder)
    data_files = ['train_xl.jsonl', 'dev.jsonl']
    for file_name in data_files:
        data_file_path = 'Data/temp/WinoGrande/winogrande_1.1/' + file_name
        assert os.path.exists(data_file_path)
        shutil.copy(data_file_path, data_folder)
    print('Data moved!')


def download_comve_data():
    print('1) Downloading ComVE...')
    temp_folder = 'Data/temp/ComVE'
    make_dir(temp_folder)
    wget.download(
        'https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/archive/refs/heads/master.zip', 
        temp_folder+'/master.zip', bar='')  #bar_adaptive)
    print('\nComVE downloaded!')
    
    print('\n2) Unpacking ComVE data...')
    with zipfile.ZipFile(temp_folder+"/master.zip","r") as zip_ref:
        zip_ref.extractall(temp_folder)
    print('Data unpacked!')
    
    print('\n3) Moving the data files...')
    data_folder = 'Data/ComVE'
    make_dir(data_folder)
    data_files = ['train.csv', 'dev.csv', 'test.csv']
    for file_name in data_files:
        data_file_path = temp_folder+'/SemEval2020-Task4-Commonsense-Validation-and-Explanation-master/ALL data/' + file_name
        assert os.path.exists(data_file_path)
        shutil.copy(data_file_path, data_folder)
    print('Data moved!')


def reformat_esnli_data():
    print('\n4) Reformatting e-SNLI data...')
    esnli_folder = 'Data/e-SNLI/'
    file1 = esnli_folder+'esnli_train_1.csv'
    file2 = esnli_folder+'esnli_train_2.csv'
    data1 = open_csv_data(file1)
    data2 = open_csv_data(file2)
    data = data1 + data2

    with open(esnli_folder+'esnli_train.csv', 'w') as write_file:
        writer = csv.DictWriter(write_file, fieldnames=list(data[0].keys()))
        print('writing...')
        writer.writeheader()
        for dat in tqdm(data):
            writer.writerow(dat)
    os.remove(file1)
    os.remove(file2)
    print('e-SNLI data is ready!')


def reformat_winogrande_data():
    print('\n4) Reformatting WinoGrande data...')
    wg_folder = 'Data/e-WG/'
    train_data_path = wg_folder + 'train_xl.jsonl'
    test_data_path = wg_folder + 'dev.jsonl'
    with jsonlines.open(train_data_path) as reader:
        train_data = list(reader)
    os.remove(train_data_path)
    with jsonlines.open(test_data_path) as reader:
        test_data = list(reader)
    
    # Create the dev dataset
    dev_size = 1268
    random.seed(2089)
    data_size = len(train_data)
    # Preserve the "pairs" of examples when doing split
    id_list = list(range(len(train_data) // 2))
    dev_ids = random.sample(id_list, dev_size // 2)
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
        
    # Create the test dataset
    random.seed(4509)
    test_size = 100
    test_data_100 = random.sample(test_data, test_size)

    # Write the new data to files
    out_file = wg_folder + 'dev.jsonl'
    with jsonlines.open(out_file, 'w') as writer:
        writer.write_all(dev_data)
    out_file = wg_folder + 'train.jsonl'
    with jsonlines.open(out_file, 'w') as writer:
        writer.write_all(new_train_data)
    out_file = wg_folder + 'test_100.jsonl'
    with jsonlines.open(out_file, 'w') as writer:
        writer.write_all(test_data_100)
    out_file = wg_folder + 'test.jsonl'
    with jsonlines.open(out_file, 'w') as writer:
        writer.write_all(test_data)
    
    # Copy (with new names) our NLE data to the Data/e-WG folder
    data_files = ['WG_train_50NLEs.jsonl', 'WG_dev_50NLEs.jsonl', 'WG_test_100NLEs.jsonl']
    new_file_names = ['train_50_nles_only.jsonl', 'dev_50_nles_only.jsonl', 'test_100_nles_only.jsonl']
    for file_name, new_file_name in zip(data_files, new_file_names):
        data_file_path = 'small-e-WinoGrande/' + file_name
        shutil.copy(data_file_path, 'Data/e-WG/')
        os.rename('Data/e-WG/'+file_name, 'Data/e-WG/'+new_file_name)
        
    # Combine the 50 NLEs from "train_50_nles_only.jsonl" with "train.jsonl" 
    #  to obtain "train_50_nles.jsonl"
    with jsonlines.open(wg_folder+'train_50_nles_only.jsonl', 'r') as reader:
        train_50_nles_only_data = list(reader)
    with jsonlines.open(wg_folder+'train.jsonl', 'r') as reader:
        train_data = list(reader)
    for idx, dat in enumerate(train_data):
        for nle_dat in train_50_nles_only_data:
            if dat['qID'] == nle_dat['qID']:
                train_data[idx] = train_50_nles_only_data[idx]
                break
    out_file = wg_folder + 'train_50_nles.jsonl'
    with jsonlines.open(out_file, 'w') as writer:
        writer.write_all(train_data)
        
    print('WinoGrande data is ready!')


def reformat_comve_data():
    print('\n4) Reformatting ComVE data...')
    comve_folder = 'Data/ComVE/'
    train_data_no_nles, _ = process_comve_data(
        comve_folder+'train.csv', random_seed=4904095, drop_nles=True)
    write_comve_data(train_data_no_nles, data_path=comve_folder+'train_no_nles.csv')

    train_data, _ = process_comve_data(comve_folder+'train.csv', random_seed=45923)

    _, selected_ids = process_comve_data(
        comve_folder+'train.csv', random_seed=215050, remove_caps_examples=True)
    os.remove(comve_folder+'train.csv')
    random.seed(4539834059)
    train_ids_1000 = random.sample(selected_ids, 1000)
    train_ids_50 = train_ids_1000[:50]
    train_nles_50 = [dat for idx, dat in enumerate(train_data)
                     if idx in train_ids_50]
    write_comve_data(train_nles_50, data_path=comve_folder+'train_50_nles_only.csv')

    train_50_nles_data = [dat if idx not in train_ids_50 else train_data[idx]
                          for idx, dat in enumerate(train_data_no_nles)]
    write_comve_data(train_50_nles_data, data_path=comve_folder+'train_50_nles.csv')

    dev_data, _ = process_comve_data(comve_folder+'dev.csv', random_seed=4354341)
    os.remove(comve_folder+'dev.csv')
    write_comve_data(dev_data, data_path=comve_folder+'dev.csv')

    test_data, _ = process_comve_data(comve_folder+'test.csv', random_seed=98750435)
    os.remove(comve_folder+'test.csv')
    write_comve_data(test_data, data_path=comve_folder+'test.csv')
    
    random.seed(5367894)
    test_ids_100 = random.sample(list(range(len(test_data))), 100)
    test_nles_100 = [dat for idx, dat in enumerate(test_data)
                       if idx in test_ids_100]
    write_comve_data(test_nles_100, data_path=comve_folder+'test_100.csv')
    print('ComVE data is ready!')


if __name__ == "__main__":
    # Frist, download all data in the correct directories via download_[]_data()
    # Second, create all data files based on the downloaded data together with the "small-e-WinoGrande" dataset
    #  via reformat_[]_data()
    print('\nI: e-SNLI\n')
    download_esnli_data()
    reformat_esnli_data()
    
    print('\n\nII: WinoGrande\n')
    download_winogrande_data()
    reformat_winogrande_data()
    
    print('\n\nIII: ComVE\n')
    download_comve_data()
    reformat_comve_data()
    
    shutil.rmtree('Data/temp')
    
    print('\nAll ready!\n')
