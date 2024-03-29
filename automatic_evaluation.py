"""
METEOR, BLEU & BERTScore evaluation for ComVE NLEs (generated by the models)
"""

import os
import csv
import random
import jsonlines
from tqdm import tqdm
from _collections import OrderedDict

import torch
from nlg_eval import get_nlg_scores, input_subset
from datasets import load_metric


print('#GPUs:', torch.cuda.device_count())
device = torch.device("cuda")
bert_metric = load_metric(
    'bertscore', experiment_id=str(random.randrange(999999)), device=device,
    cache_dir="huggingface_cache")

model_name_map_dict = OrderedDict({
    'saved_models/ComVEft50NLEs/26': 'CD--fine-tune',
    'saved_models/ComVE_NLEs/19': 'CD--union',
    'saved_models/ESNLI+SNLIftComVE/16': 'WT5--fine-tune',
    'saved_models/ESNLI_SNLI+ComVE/11': 'WT5',

    'saved_models/ESNLI_ComVE_2/11': 'M1',
    'saved_models/ESNLI_ComVEft50NLEs/13': 'M2',
    'saved_models/ESNLIftComVE_NLEs/1': 'M3',
    'saved_models/ESNLIftComVEft50NLEs/13': 'M4'
})

test_file = 'Data/ComVE/test.csv'
predictions_file_name = 'predictions_full_test.jsonl'
# Load the test data
with open(test_file, 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    test_data = list(reader)
print('Test data size:', len(test_data))
gt_labels = [int(dat['Correct option']) for dat in test_data]
gt_nles = [[dat['Right Reason1'], dat['Right Reason2'], dat['Right Reason3']]
           for dat in test_data]

scores = {}
for model_folder in tqdm(model_name_map_dict.keys()):
    model = model_name_map_dict[model_folder]
    predictions_path = os.path.join(model_folder, predictions_file_name)
    with jsonlines.open(predictions_path) as reader:
        prediction_data = list(reader)
    predicted_labels = [int(dat['answer'][:1])-1 for dat in prediction_data]
    predicted_nles = [dat['nle'] for dat in prediction_data]

    # Filter the NLEs
    selected_gt_nles, selected_predicted_nles = input_subset(
        gt_labels=gt_labels, predicted_labels=predicted_labels,
        gt_nles=gt_nles, predicted_nles=predicted_nles)

    # Get scores
    scores[model_folder] = get_nlg_scores(
        gen_expl=selected_predicted_nles, gt_expl=selected_gt_nles,
        bert_metric=bert_metric, device=device)
    # print('Model {} F1 BERTScore: {}'.format(model, scores[model_folder]))

print('\n')
score_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", 'METEOR', 'BERTScore']
print('Model & B-1 & B-2 & B-3 & B-4 & METEOR & BERTScore')
for model_folder in model_name_map_dict.keys():
    full_results_list = [
        round(scores[model_folder][name]*100, 1) for name in score_names]
    print('{} & {} & {} & {} & {} & {} & {} \\\\'.format(
        model_name_map_dict[model_folder],
        *tuple(full_results_list)))

print('\nNote: these results are against the following dataset: {}!\n'.format(test_file))

