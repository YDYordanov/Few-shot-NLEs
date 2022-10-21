"""
Datasets, DataLoaders, and helper functions
"""
import os
import csv
import jsonlines
import time

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class ESNLIDataset(Dataset):
    """
    e-SNLI dataset for WT5
    """

    def __init__(self, data_path, tokenizer, input_length=10):
        super(ESNLIDataset, self).__init__()
        self.data_path = data_path
        self.input_length = input_length
        self.tokenizer = tokenizer

        # Load the data
        data_file = os.path.join(self.data_path)
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            data = list(reader)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        premise = sample['Sentence1']
        hypothesis = sample['Sentence2']
        explanation = sample['Explanation_1']
        relation = sample['gold_label']

        if explanation == '':
            input_text = 'nli premise: {} hypothesis: {}'.format(
                premise, hypothesis)
            target_text = relation
        else:
            input_text = 'explain nli premise: {} hypothesis: {}'.format(
                premise, hypothesis)
            target_text = '{} explanation: {}'.format(relation, explanation)

        return_dict = {
            'input_text': input_text,
            'target_text': target_text
        }
        return return_dict


class WinogradDataset(Dataset):
    """
    Winograd Schema Dataset master class
    """
    def __init__(self, tokenizer, input_length=10,
                 force_explain=False, input_format='default'):
        super(WinogradDataset, self).__init__()
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.force_explain = force_explain
        self.input_format = input_format

        assert input_format in ['default', 't5', 'nli', 'no_options',
                                'correct_option_prompt']

    def process_example(self, schema_sentence, option1, option2,
                        correct_option, explanation):
        if self.input_format == 'nli':
            if schema_sentence[-1] != '.':
                print('Faulty schema:', schema_sentence)
            if schema_sentence.find('.') < len(schema_sentence) - 2:
                full_stop_id = schema_sentence.find('.')
                hypothesis = schema_sentence[full_stop_id+1:]
            elif ' because ' in schema_sentence:
                str_id = schema_sentence.find(' because ')
                hypothesis = schema_sentence[str_id+len(' because '):]
            elif ' since ' in schema_sentence:
                str_id = schema_sentence.find(' since ')
                hypothesis = schema_sentence[str_id + len(' since '):]
            elif ' so ' in schema_sentence:
                str_id = schema_sentence.find(' so ')
                hypothesis = schema_sentence[str_id+len(' so '):]
            else:
                pron_id = schema_sentence.find('_')
                hypothesis = schema_sentence[pron_id:]
            if '_' not in hypothesis:
                pron_id = schema_sentence.find('_')
                hypothesis = schema_sentence[pron_id:]
            hypothesis = hypothesis.replace('_', option1).strip()
            hypothesis = hypothesis.capitalize()
            input_text = 'nli premise: {} hypothesis: {}'.format(
                schema_sentence, hypothesis)
        elif self.input_format == 't5':
            schema_sentence = schema_sentence.replace("_", '<extra_id_0>')
            input_text = 'schema: {} options: {}, {}.'.format(
                schema_sentence, option1, option2)
        elif self.input_format == 'no_options':
            input_text = 'schema: {}'.format(schema_sentence)
        elif self.input_format == 'correct_option_prompt':
            input_text = 'schema: {} options: {}, {}. correct option:'.format(
                schema_sentence, option1, option2)
        else:
            assert self.input_format == 'default'
            input_text = 'schema: {} options: {}, {}.'.format(
                schema_sentence, option1, option2)

        if explanation is not None or self.force_explain:
            input_text = 'explain ' + input_text

        if self.input_format == 'nli':
            target_text = ('entailment' if option1 == correct_option
                           else 'contradiction')
        else:
            target_text = correct_option
            if self.input_format == 't5':
                target_text = '<extra_id_0> ' + target_text + ' <extra_id_1>'

        if explanation is not None:
            target_text += ' explanation: {}'.format(explanation)

        return_dict = {
            'input_text': input_text,
            'target_text': target_text,
            'option1': option1,
            'option2': option2
        }
        return return_dict


class EWGDataset(WinogradDataset):
    """
    e-WG dataset for WT5
    """
    def __init__(self, data_path, tokenizer, input_length=10,
                 force_explain=False, input_format='default'):
        super(EWGDataset, self).__init__(
            tokenizer, input_length=input_length,
            force_explain=force_explain, input_format=input_format)
        self.data_path = data_path
        # Load the data
        data_file = os.path.join(self.data_path)
        with jsonlines.open(data_file) as reader:
            data = list(reader)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert type(sample) == dict

        schema_sentence = sample['sentence']
        option1 = sample['option1']
        option2 = sample['option2']
        answer = int(sample['answer'])
        correct_option = (option1 if answer == 1 else option2)
        if 'nle' in sample.keys():
            explanation = sample['nle']
        else:
            explanation = None

        return_dict = self.process_example(
            schema_sentence=schema_sentence, option1=option1,
            option2=option2, correct_option=correct_option,
            explanation=explanation)
        return return_dict


class ComVEDataset(Dataset):
    """
    ComVE Dataset for WT5
    """

    def __init__(self, data_path,  tokenizer, force_explain=False):
        super(ComVEDataset, self)
        self.data_path = data_path
        self.force_explain = force_explain
        self.tokenizer = tokenizer
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            data = list(reader)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert type(sample) == dict

        sent1 = sample['Sentence1']
        sent2 = sample['Sentence2']
        correct_option_id = sample['Correct option']

        explanation = sample['Right Reason1']

        input_text = 'ComVE Sentence 1: {} Sentence 2: {}'.format(
            sent1, sent2)

        if explanation is not None or self.force_explain:
            input_text = 'explain ' + input_text

        target_text = str(int(correct_option_id) + 1)
        if explanation is not None:
            target_text += ' explanation: {}'.format(explanation)

        return_dict = {
            'input_text': input_text,
            'target_text': target_text
        }
        return return_dict


class ESNLIEWGDataset(WinogradDataset):
    """
    e-SNLI+e-WG dataset for WT5
    """
    def __init__(self, wg_data_path, snli_data_path, tokenizer,
                 input_length=10, force_explain=False, input_format='default'):
        super(ESNLIEWGDataset, self).__init__(
            tokenizer, input_length=input_length,
            force_explain=force_explain, input_format=input_format)
        self.wg_data_path = wg_data_path
        self.snli_data_path = snli_data_path
        # Load the data
        data_file = os.path.join(self.wg_data_path)
        with jsonlines.open(data_file) as reader:
            wg_data = list(reader)
        data_file = os.path.join(self.snli_data_path)
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            snli_data = list(reader)
        self.data = wg_data + snli_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert type(sample) == dict

        if 'gold_label' in sample.keys():
            premise = sample['Sentence1']
            hypothesis = sample['Sentence2']
            input_text = 'explain nli premise: {} hypothesis: {}'.format(
                premise, hypothesis)

            relation = sample['gold_label']
            explanation = sample['Explanation_1']
            target_text = '{} explanation: {}'.format(relation, explanation)

            return_dict = {
                'input_text': input_text,
                'target_text': target_text
            }
        else:
            schema_sentence = sample['sentence']
            option1 = sample['option1']
            option2 = sample['option2']
            answer = int(sample['answer'])
            correct_option = (option1 if answer == 1 else option2)
            if 'nle' in sample.keys():
                explanation = sample['nle']
            else:
                explanation = None

            return_dict = self.process_example(
                schema_sentence=schema_sentence, option1=option1,
                option2=option2, correct_option=correct_option,
                explanation=explanation)
        return return_dict


class ESNLIComVEDataset(WinogradDataset):
    """
    e-SNLI+ComVE dataset for WT5
    """
    def __init__(self, comve_data_path, snli_data_path, tokenizer,
                 input_length=10, force_explain=False, input_format='default'):
        super(ESNLIComVEDataset, self).__init__(
            tokenizer, input_length=input_length,
            force_explain=force_explain, input_format=input_format)
        self.comve_data_path = comve_data_path
        self.snli_data_path = snli_data_path
        # Load the data
        data_file = os.path.join(self.comve_data_path)
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            comve_data = list(reader)
        data_file = os.path.join(self.snli_data_path)
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            snli_data = list(reader)
        self.data = comve_data + snli_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        assert type(sample) == dict

        if 'gold_label' in sample.keys():
            premise = sample['Sentence1']
            hypothesis = sample['Sentence2']
            input_text = 'explain nli premise: {} hypothesis: {}'.format(
                premise, hypothesis)

            relation = sample['gold_label']
            explanation = sample['Explanation_1']
            target_text = '{} explanation: {}'.format(relation, explanation)

            return_dict = {
                'input_text': input_text,
                'target_text': target_text
            }
        else:
            sent1 = sample['Sentence1']
            sent2 = sample['Sentence2']
            correct_option_id = sample['Correct option']

            explanation = sample['Right Reason1']

            input_text = 'ComVE Sentence 1: {} Sentence 2: {}'.format(
                sent1, sent2)

            if explanation is not None or self.force_explain:
                input_text = 'explain ' + input_text

            target_text = str(int(correct_option_id) + 1)
            if explanation is not None:
                target_text += ' explanation: {}'.format(explanation)

            return_dict = {
                'input_text': input_text,
                'target_text': target_text
            }
        return return_dict


class WT5Loader:
    def __init__(self, data_paths, tokenizer, input_length, batch_size,
                 task_name='esnli', do_train=False, force_explain=False,
                 input_format='default'):
        if isinstance(data_paths, list):
            self.data_paths = data_paths
        elif isinstance(data_paths, str):
            self.data_paths = [data_paths]
        else:
            raise TypeError
        self.input_length = input_length
        self.batch_size = batch_size
        self.do_train = do_train
        self.task_name = task_name

        if self.task_name == 'esnli':
            self.dataset = ESNLIDataset(
                data_path=self.data_paths[0],
                tokenizer=tokenizer,
                input_length=self.input_length
            )
        elif self.task_name == 'ewg':
            self.dataset = EWGDataset(
                data_path=self.data_paths[0],
                tokenizer=tokenizer,
                input_length=self.input_length,
                force_explain=force_explain,
                input_format=input_format
            )
        elif self.task_name in ['wg_acc', 'wg_nles']:
            self.dataset = EWGDataset(
                data_path=self.data_paths[0],
                tokenizer=tokenizer,
                input_length=self.input_length,
                force_explain=False,
                input_format=input_format
            )
        elif self.task_name == 'comve':
            self.dataset = ComVEDataset(
                data_path=self.data_paths[0],
                tokenizer=tokenizer,
                force_explain=force_explain
            )
        elif self.task_name in ['comve_acc', 'comve_nles']:
            self.dataset = ComVEDataset(
                data_path=self.data_paths[0],
                tokenizer=tokenizer,
                force_explain=False
            )
        elif self.task_name == 'esnli+ewg':
            self.dataset = ESNLIEWGDataset(
                wg_data_path=self.data_paths[1],
                snli_data_path=self.data_paths[0],
                tokenizer=tokenizer,
                input_length=self.input_length,
                input_format=input_format,
                force_explain=force_explain
            )
        elif self.task_name == 'esnli+comve':
            self.dataset = ESNLIComVEDataset(
                comve_data_path=self.data_paths[1],
                snli_data_path=self.data_paths[0],
                tokenizer=tokenizer,
                input_length=self.input_length,
                input_format=input_format,
                force_explain=force_explain
            )
        else:
            raise NotImplementedError

        def collate_fn(input_list):
            input_texts = [entry['input_text'] for entry in input_list]
            target_texts = [entry['target_text'] for entry in input_list]
            input_dict = tokenizer(
                input_texts, return_tensors='pt', padding=True)
            input_ids = input_dict.input_ids
            # input_mask = input_dict.attention_mask
            target_dict = tokenizer(
                target_texts, return_tensors='pt', padding=True)
            target_ids = target_dict.input_ids
            # target_mask = target_dict.attention_mask
            return_dict = {
                'input_tensor': input_ids,
                'target_tensor': target_ids
            }

            # These options are to constrain the beam search later on
            for entry in input_list:
                for dict_key in entry.keys():
                    if 'option' in dict_key:
                        if dict_key in return_dict.keys():
                            return_dict[dict_key].append(entry[dict_key])
                        else:
                            return_dict[dict_key] = [entry[dict_key]]
            for dict_key in return_dict.keys():
                if 'option' in dict_key:
                    return_dict[dict_key] = tokenizer(
                        return_dict[dict_key], return_tensors='pt',
                        padding=True).input_ids

            return return_dict

        if self.do_train:
            self.data_loader = DataLoader(
                self.dataset, batch_size=self.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=8, pin_memory=True,
                worker_init_fn=np.random.seed(2809))
        else:
            self.data_loader = DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          collate_fn=collate_fn)
        self.data_loader.task_name = task_name


def pad_sentences(sent_list, max_length, pad_idx=2, mask=None):
    b_size = len(sent_list)
    sent_lens = [len(sent) for sent in sent_list]
    max_sent_len = max(sent_lens)
    assert max_sent_len <= max_length

    def pad_x(x_):
        # Pad the data entry (list) to max_input_length (pad_id=-1)
        x_ = x_ + [pad_idx for _ in range(max_sent_len - len(x_))]
        x_ = torch.LongTensor(x_).unsqueeze(0)
        return x_

    padded_sentences = torch.cat([pad_x(sent) for sent in sent_list])
    return padded_sentences
