import os
import json
from _collections import OrderedDict


ewg_model_name_map_dict = OrderedDict({
    'saved_models/WGft50NLEs_reproduced/25': 'CD--fine-tune',
    'saved_models/WG+50NLEs_reproduced/22': 'CD--union',
    'saved_models/ESNLI+SNLIftWG_reproduced/19': 'WT5--fine-tune',
    'saved_models/ESNLI+SNLI+WG_reproduced/11': 'WT5',
    
    'saved_models/ESNLI+WG+50NLEs_reproduced/18': 'M1',
    'saved_models/ESNLI+WGft50NLEs_reproduced/16': 'M2',
    'saved_models/ESNLIftWG+50NLEs_reproduced/11': 'M3',
    'saved_models/ESNLIftWGft50NLEs_reproduced/22': 'M4'
})

comve_model_name_map_dict = OrderedDict({
    'saved_models/ComVEft50NLEs_reproduced/26': 'CD--fine-tune',
    'saved_models/ComVE+50NLEs_reproduced/19': 'CD--union',
    'saved_models/ESNLI+SNLIftComVE_reproduced/16': 'WT5--fine-tune',
    'saved_models/ESNLI+SNLI+ComVE_reproduced/11': 'WT5',
    
    'saved_models/ESNLI+ComVE+50NLEs_reproduced/11': 'M1',
    'saved_models/ESNLI+ComVEft50NLEs_reproduced/13': 'M2',
    'saved_models/ESNLIftComVE+50NLEs_reproduced/1': 'M3',
    'saved_models/ESNLIftComVEft50NLEs_reproduced/13': 'M4'
})

model_name_map_dicts = [ewg_model_name_map_dict, comve_model_name_map_dict]

metrics_batches = [
    ['ewg_acc'],
    ['comve_acc']]
print('Metrics:', metrics_batches, '\n')

headers = [
    'Models & WG acc\% \\\\',
    'Models & ComVE acc\% \\\\'
]

for header, model_name_map_dict, metrics in zip(
        headers, model_name_map_dicts, metrics_batches):
    print('\n')
    print(header)
    print('\\hline')
    for model_folder in model_name_map_dict.keys():
        metric_values = []
        model_name = model_name_map_dict[model_folder]
        for metric in metrics:
            if metric == 'esnli_ppl':
                metric_file_name = 'esnli_test_results.json'
                metric_name = 'nle_perplexity'
            elif metric == 'esnli_acc':
                metric_file_name = 'esnli_test_results.json'
                metric_name = 'Accuracy'
            elif metric == 'ewg_acc':
                metric_file_name = 'ewg_test_results.json'
                metric_name = 'Accuracy'
            elif metric == 'ewg_dev_ppl':
                metric_file_name = 'final_wg_nles_dev_results.json'
                metric_name = 'nle_perplexity'
            elif metric == 'comve_acc':
                metric_file_name = 'comve_test_results.json'
                metric_name = 'Accuracy'
            elif metric == 'comve_test_ppl':
                metric_file_name = 'comve_test_results.json'
                metric_name = 'nle_perplexity'
            else:
                raise NotImplementedError

            metric_file_path = os.path.join(model_folder, metric_file_name)
            if os.path.exists(metric_file_path):
                with open(metric_file_path, 'r') as f:
                    data = json.load(f)
                if metric_name == 'Accuracy':
                    metric_value = str(round(data[metric_name]*100, 1))
                else:
                    metric_value = round(data[metric_name], 2)
            else:
                metric_value = 'n/a'
            metric_values.append(metric_value)

        print('{} & {} \\\\'.format(
            model_name, *tuple(metric_values)
            )
        )
    print('')

