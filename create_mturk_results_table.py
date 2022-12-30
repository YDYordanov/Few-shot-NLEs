"""
Extract and visualize the results from MTurk
"""
import os
import csv
import argparse
from _collections import OrderedDict
from collections import Counter
import random


def print_hit_example(hit, ex_id_in_hit, shortcoming_options):
    print('Reviewer', hit['WorkerId'],
          'Total time spent:', hit['WorkTimeInSeconds'])
    if 'Input.schema_{}'.format(ex_id_in_hit) in hit.keys():
        print(hit['Input.schema_{}'.format(ex_id_in_hit)])
        option1 = hit['Input.option_1_{}'.format(ex_id_in_hit)]
        option2 = hit['Input.option_2_{}'.format(ex_id_in_hit)]
        print('Option 1:', option1)
        print('Option 2:', option2)
    else:
        print('Statement 1:', hit['Input.statement_1_{}'.format(ex_id_in_hit)])
        print('Statement 2:', hit['Input.statement_2_{}'.format(ex_id_in_hit)])
        option1 = 'statement 1'
        option2 = 'statement 2'

    gt = hit['Input.gt_{}'.format(ex_id_in_hit)]
    options = [option1, option2]
    print('Correct:', options[int(gt)])
    print('NLE 1:', hit['Input.nle_1_{}'.format(ex_id_in_hit)])
    print('Score:', hit['Answer.e1_{}'.format(ex_id_in_hit)])
    print('Shortcomings:')
    shortcomings = hit[
        'Answer.f1_{}'.format(ex_id_in_hit)].replace('|', '')
    for shortcoming in shortcomings:
        print('  ', shortcoming_options[int(shortcoming)])
    print('NLE 2:', hit['Input.nle_2_{}'.format(ex_id_in_hit)])
    print('Score:', hit['Answer.e2_{}'.format(ex_id_in_hit)])
    print('Shortcomings:')
    shortcomings = hit[
        'Answer.f2_{}'.format(ex_id_in_hit)].replace('|', '')
    for shortcoming in shortcomings:
        print('  ', shortcoming_options[int(shortcoming)])
    print('------------')


def get_model_path(model_name_map_dict, model_name='(PT, CT, eCT50)', comve=True):
    model_paths = []
    for key in model_name_map_dict.keys():
        if model_name == model_name_map_dict[key]:
            model_paths.append(key)
    if comve:
        model_paths = [path for path in model_paths if 'ComVE' in path]
    else:
        model_paths = [path for path in model_paths if
                       ('WG' in path or 'BIG_T5' in path)]
    assert len(model_paths) == 1
    return model_paths[0]


def extract_mturk_results(
        results_paths, ignore_annotators: list, visualize, print_annotator):
    # Number of examples per HIT
    num_examples = 10
    annotators_per_example = 3

    data = []
    for batch_result_path in results_paths:
        # batch_result_path
        print('Extracting result file {} ...'.format(batch_result_path))
        # Open the batch results file
        with open(batch_result_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            data += list(reader)
    # Ignore some annotators
    other_hits = []
    for hit in data:
        if hit['WorkerId'] not in ignore_annotators:
            other_hits += [hit]
    data = other_hits

    # Check for repeating reviewers
    # Select only the annotator's HITs as multi-index
    data_multi_ids = []
    for hit in data:
        data_multi_ids += [
            {'multi_id': [{
                'Input.model_{}'.format(i):
                    hit['Input.model_{}'.format(i)],
                'Input.example_id_{}'.format(i):
                    hit['Input.example_id_{}'.format(i)]
                } for i in range(10)],
            'worker_id': hit['WorkerId']
            }]
    multi_id_set = []
    for dat in data_multi_ids:
        if dat['multi_id'] not in multi_id_set:
            multi_id_set += dat['multi_id']
    for multi_id in multi_id_set:
        worker_set = [
            dat['worker_id'] for dat in data_multi_ids
            if dat['multi_id'] == multi_id]
        if len(worker_set) != len(set(worker_set)):
            print('Warning: repeating annotator!')

    shortcoming_options = [
        'Does not make sense',
        'Insufficient justification',
        'Irrelevant to the schema',
        'Too trivial',
        'None'
    ]

    if visualize:
        for hit in data:
            for ex_id_in_hit in range(num_examples):
                print_hit_example(
                    hit, ex_id_in_hit=ex_id_in_hit, shortcoming_options=shortcoming_options)

        # Print a random sample as well!
        num_examples_annotator = len(data) * num_examples
        # take a random sample of 10 NLEs
        random.seed(4583)
        sampled_example_ids = random.sample(
            list(range(num_examples_annotator)), 10)
        print('\n------------\n')
        for multi_id in sampled_example_ids:
            hit_id = multi_id // num_examples
            ex_id_in_hit = multi_id % num_examples
            print_hit_example(
                hit=data[hit_id], ex_id_in_hit=ex_id_in_hit,
                shortcoming_options=shortcoming_options)

    # First, extract the model names from the data:
    model_names = []
    for hit in data:
        for ex_id_in_hit in range(num_examples):
            model_name = hit['Input.model_{}'.format(ex_id_in_hit)]
            if model_name not in model_names:
                model_names += [model_name]

    # For each model name, find its example ids
    example_ids = {}
    for model in model_names:
        example_ids[model] = []
        for hit in data:
            for ex_id_in_hit in range(num_examples):
                if hit['Input.model_{}'.format(ex_id_in_hit)] == model:
                    example_id = hit[
                        'Input.example_id_{}'.format(ex_id_in_hit)]
                    if example_id not in example_ids[model]:
                        example_ids[model] += [example_id]
        assert example_ids[model] != []

    results = {}
    nles = {}
    worker_stats = {}
    schemas = {}
    shortcomings = {}
    annotation_counts = {}
    hits = {}
    for mod in model_names:
        annotation_counts[mod] = {}
        for ex_id in range(100):
            annotation_counts[mod][str(ex_id)] = 0

    for model in model_names:
        results[model] = []
        nles[model] = []
        worker_stats[model] = []
        schemas[model] = []
        shortcomings[model] = []
        hits[model] = []
        for example_id in example_ids[model]:
            for hit in data:
                for ex_id_in_hit in range(num_examples):
                    if (
                            hit['Input.model_{}'.format(ex_id_in_hit)] == model
                            and
                            hit['Input.example_id_{}'.format(
                                ex_id_in_hit)] == example_id
                            and
                            annotation_counts[
                                model][example_id] < annotators_per_example
                    ):
                        answers = [
                            hit['Answer.e1_{}'.format(ex_id_in_hit)],
                            hit['Answer.e2_{}'.format(ex_id_in_hit)]
                        ]
                        answer = answers[
                            1 - int(hit['Input.gt_nle_{}'.format(
                                ex_id_in_hit)])]
                        results[model] += [answer]
                        nles[model] += [
                            hit['Input.nle_{}_{}'.format(
                                2 - int(hit['Input.gt_nle_{}'.format(ex_id_in_hit)]),
                                ex_id_in_hit)]]
                        worker_stats[model] += [
                            [hit['WorkerId'], hit['WorkTimeInSeconds']]
                        ]
                        hits[model] += [hit]

                        schema_key = 'Input.schema_{}'.format(ex_id_in_hit)
                        option_keys = [
                            'Input.statement_1_{}'.format(ex_id_in_hit),
                            'Input.statement_2_{}'.format(ex_id_in_hit)
                        ]
                        if schema_key in hit.keys():
                            schemas[model] += [
                                hit[schema_key]]
                        else:
                            schemas[model] += [
                                [hit[option_keys[0]], hit[option_keys[1]]]]

                        shortcoming_pair = [
                            hit['Answer.f1_{}'.format(ex_id_in_hit)].replace(
                                '|', ''),
                            hit['Answer.f2_{}'.format(ex_id_in_hit)].replace(
                                '|', '')
                        ]
                        shortcoming = shortcoming_pair[
                            1 - int(hit['Input.gt_nle_{}'.format(
                                ex_id_in_hit)])]

                        # Note: here we split the shortcomings of each example
                        shortcomings[model] += [
                            [shortcoming_options[int(char)] for char in shortcoming]]

                        annotation_counts[model][example_id] += 1

    # Get the annotator set with:
    # 1) annotation counts
    # 2) annotation time
    # 3) score distribution (yes/w_yes/w_no/no%)
    all_workers = {}
    for model in model_names:
        for example_id in example_ids[model]:
            for hit in data:
                for ex_id_in_hit in range(num_examples):
                    if (
                            hit['Input.model_{}'.format(ex_id_in_hit)] == model
                            and
                            hit['Input.example_id_{}'.format(
                                ex_id_in_hit)] == example_id
                    ):
                        gt = hit['Input.gt_{}'.format(ex_id_in_hit)]

                        answers = [
                            hit['Answer.e1_{}'.format(ex_id_in_hit)],
                            hit['Answer.e2_{}'.format(ex_id_in_hit)]
                        ]
                        answer = answers[
                            1 - int(hit['Input.gt_nle_{}'.format(
                                ex_id_in_hit)])]
                        gt_answer = answers[
                            int(hit['Input.gt_nle_{}'.format(
                                ex_id_in_hit)])]

                        # Add shortcomings statistics, in %
                        shortcomings_ = hit[
                            'Answer.f2_{}'.format(ex_id_in_hit)
                        ].replace('|', '')
                        list_shortcomings = [
                            shortcoming_options[int(short)] for short in shortcomings_]

                        if hit['WorkerId'] in all_workers.keys():
                            all_workers[hit['WorkerId']]['#hits'] += 1
                            all_workers[hit['WorkerId']]['times'] += [
                                int(hit['WorkTimeInSeconds'])]
                            all_workers[hit['WorkerId']]['scores'] += [answer]
                            all_workers[hit['WorkerId']]['gt_scores'] += [
                                gt_answer]
                            all_workers[hit['WorkerId']]['shortcomings'] += [
                                list_shortcomings]

                            if 'Answer.c{}'.format(ex_id_in_hit) in hit.keys():
                                label = hit['Answer.c{}'.format(ex_id_in_hit)]
                                if gt == label:
                                    all_workers[hit['WorkerId']]['correct'] += 1
                            else:
                                pass
                                # print('Warning: Answer.c{} not found!'.format(
                                #     ex_id_in_hit))
                            all_workers[hit['WorkerId']]['num_all'] += 1
                        else:
                            all_workers[hit['WorkerId']] = {}
                            all_workers[hit['WorkerId']]['#hits'] = 1
                            all_workers[hit['WorkerId']]['times'] = [
                                int(hit['WorkTimeInSeconds'])]
                            all_workers[hit['WorkerId']]['scores'] = [answer]
                            all_workers[hit['WorkerId']]['gt_scores'] = [
                                gt_answer]
                            all_workers[hit['WorkerId']]['shortcomings'] = [
                                list_shortcomings]
                            all_workers[hit['WorkerId']]['correct'] = 0
                            all_workers[hit['WorkerId']]['num_all'] = 1

    for worker_id in all_workers.keys():
        all_workers[worker_id]['times'] = [
            min(all_workers[worker_id]['times']),
            max(all_workers[worker_id]['times']),
            round(sum(all_workers[worker_id]['times']) / len(
                all_workers[worker_id]['times']), 1)
        ]
        num_scores = len(all_workers[worker_id]['scores'])
        scores = all_workers[worker_id]['scores']
        all_workers[worker_id]['scores'] = {
            'yes': round(scores.count('yes') / num_scores, 2),
            'w_yes': round(scores.count('w_yes') / num_scores, 2),
            'w_no': round(scores.count('w_no') / num_scores, 2),
            'no': round(scores.count('no') / num_scores, 2)
        }
        gt_scores = all_workers[worker_id]['gt_scores']
        all_workers[worker_id]['gt_scores'] = {
            'yes': round(gt_scores.count('yes') / num_scores, 2),
            'w_yes': round(gt_scores.count('w_yes') / num_scores, 2),
            'w_no': round(gt_scores.count('w_no') / num_scores, 2),
            'no': round(gt_scores.count('no') / num_scores, 2)
        }
        all_workers[worker_id]['correct'] = round(
            100 * all_workers[worker_id]['correct'] /
            all_workers[worker_id]['num_all'], 1)
        # add statistics of the shortcomings
        shortcomings_list = all_workers[worker_id]['shortcomings']
        shortcomings_dict = {checkbox: 0 for checkbox in shortcoming_options}
        for short_list in shortcomings_list:
            for short in short_list:
                shortcomings_dict[short] += 1
        shortcomings_dict = {
            short: round(
                100 * shortcomings_dict[short] /
                sum(shortcomings_dict.values()), 1)
            for short in shortcomings_dict.keys()}
        all_workers[worker_id]['shortcomings'] = shortcomings_dict
        
    # Sort the workers by how many HITs they did
    all_workers = {k: v for k, v in sorted(
        all_workers.items(), key=lambda item: item[1]['#hits'], reverse=True)}
    for worker_id in all_workers.keys():
        if visualize:
            print(worker_id, all_workers[worker_id])
    if visualize:
        print('------------')
        print('Top workers:')
    # Filter-out only the whales:
    whale_workers = {}
    for worker_id, worker_info in all_workers.items():
        if worker_info['#hits'] >= 50:
            whale_workers[worker_id] = worker_info
    for worker_id in whale_workers.keys():
        if visualize:
            print(worker_id, whale_workers[worker_id])
    print('Number of workers:', len(list(all_workers.keys())))
    print('Cost:', sum([int(all_workers[key]['#hits'])
                        for key in all_workers.keys()]) / 10)

    # Print a random sample of NLEs from a given reviewer
    if print_annotator != '':
        # select all HITs for this annotator
        annotator_hits = []
        for hit in data:
            if hit['WorkerId'] == print_annotator:
                annotator_hits += [hit]
        num_examples_annotator = len(annotator_hits) * num_examples
        # take a random sample of 10 NLEs
        random.seed(4583)
        sampled_example_ids = random.sample(
            list(range(num_examples_annotator)), 10)
        print('-----------')
        for multi_id in sampled_example_ids:
            hit_id = multi_id // num_examples
            ex_id_in_hit = multi_id % num_examples
            print_hit_example(
                hit=annotator_hits[hit_id], ex_id_in_hit=ex_id_in_hit,
                shortcoming_options=shortcoming_options)

    nle_labels = ['yes', 'w_yes', 'w_no', 'no']
    results_dict = {}
    for model in results.keys():
        results_dict[model] = {}
        for label in nle_labels:
            num_labels = sum([res == label for res in results[model]])
            results_dict[model][label] = num_labels

    # Map each model dir to the model name as in the table
    model_name_map_dict = OrderedDict({
        'saved_models/WGftWG-nles_fixed/25': 'CT--fine-tune',
        'saved_models/EWG_6/22': 'CT--union',
        'saved_models/ESNLI+SNLIftWG/19': 'WT5--fine-tune',
        'saved_models/ESNLI_SNLI+WG/11': 'WT5',

        'saved_models/ESNLI_EWG_9/18': 'M1',
        'saved_models/ESNLI+WGftWG-nles_fixed/16': 'M2',
        'saved_models/ESNLIftEWG/11': 'M3',
        'saved_models/ESNLIftWGftWG-nles_fixed/22': 'M4',

        'saved_models/ComVEft50NLEs/26': 'CT--fine-tune',
        'saved_models/ComVE_NLEs/19': 'CT--union',
        'saved_models/ESNLI+SNLIftComVE/16': 'WT5--fine-tune',
        'saved_models/ESNLI_SNLI+ComVE/11': 'WT5',

        'saved_models/ESNLI_ComVE_2/11': 'M1',
        'saved_models/ESNLI_ComVEft50NLEs/13': 'M2',
        'saved_models/ESNLIftComVE_NLEs/1': 'M3',
        'saved_models/ESNLIftComVEft50NLEs/13': 'M4',
    })

    # Get the post-filtered annotator set
    all_workers = {}
    for model in model_name_map_dict.keys():
        if model in results_dict.keys():
            for example_id in example_ids[model]:
                # Assert that we have selected the correct files
                assert annotation_counts[model][example_id] == 3

                for hit in data:
                    for ex_id_in_hit in range(num_examples):
                        if (
                                hit['Input.model_{}'.format(ex_id_in_hit)] == model
                                and
                                hit['Input.example_id_{}'.format(
                                    ex_id_in_hit)] == example_id
                        ):
                            if hit['WorkerId'] in all_workers.keys():
                                all_workers[hit['WorkerId']] += 1
                            else:
                                all_workers[hit['WorkerId']] = 1
    print('Filtered number of workers:', len(list(all_workers.keys())))
    print('Filtered cost:', sum([int(all_workers[key]) for key in all_workers.keys()]) / 10)
    
    # Create the table with NLE human evaluation. Normalize by sum of scores.
    print('\nTable 5 of the Appendix:')
    print('\n\\textbf{Models} & \\textbf{NLE score} & \\textbf{Yes\%} & \\textbf{Weak Yes\%} & '
          '\\textbf{Weak No\%} & \\textbf{No\%} & '
          '\\textbf{Does not make sense\%} & '
          '\\textbf{Insufficient justification\%} & '
          '\\textbf{Irrelevant to the schema\%} & '
          '\\textbf{Too trivial\%} & '
          '\\textbf{None\%}\\\\')
    print('\\hline')
    for model in model_name_map_dict.keys():
        if model in results_dict.keys():
            model_name = model_name_map_dict[model]
            result_dict = results_dict[model]
            result_list = [result_dict[label] for label in nle_labels]
            result_sum = sum(result_list)
            num_correct = result_sum / annotators_per_example
            assert num_correct == int(num_correct)
            num_correct = int(num_correct)
            full_result_list = []
            for res in result_list:
                full_result_list += [round(res / result_sum * 100, 1)]
                
            # Calculate the NLE score
            agg_result = round(
                (result_dict['yes'] + (2 / 3) * result_dict['w_yes'] +
                 (1 / 3) * result_dict['w_no']
                 ) / result_sum * 100, 1)
                 
            # Calculate the "shortcomings"
            bag_of_words = Counter(sum(shortcomings[model], []))
            shortcoming_counts = [
                bag_of_words[shortcoming]
                for shortcoming in shortcoming_options]
            shortcoming_percentages = [
                round(100 * count_ / sum(shortcoming_counts), 1)
                for count_ in shortcoming_counts]
                
            print('{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(
                    model_name,
                    agg_result, *tuple(full_result_list), *tuple(shortcoming_percentages)
                )
            )
          
    return results_dict, model_name_map_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers model")
    parser.add_argument('--visualize', '-v', action='store_true',
                        help="Visualize the results from the MT workers.")
    parser.add_argument('--print_sample_from_annotator', '-print_annotator',
                        type=str, default='',
                        help="Print a sample of annotations from a "
                             "given annotator")
    args = parser.parse_args()
        
    print('\nWinoGrande results:\n')
    winogrande_results_paths = [
        'Results/WG_results.csv',
        'Results/wg_results_re_annotating_A35D6EOL7V8U03.csv']
    extract_mturk_results(
        results_paths=winogrande_results_paths,
        ignore_annotators=['A35D6EOL7V8U03'],
        visualize=args.visualize,
        print_annotator=args.print_sample_from_annotator)
        
    print('\nComVE results:\n')
    comve_results_paths = [
        'Results/ComVE_results.csv', 
        'Results/comve_results_re_annotating_A3PYK6HSL97EFU.csv', 
        'Results/comve_results_re_annotating_ANR0LIX0VLUYJ.csv', 
        'Results/Batch_4440741_batch_results(ComVE_patch).csv']
    extract_mturk_results(
        results_paths=comve_results_paths,
        ignore_annotators=['ANR0LIX0VLUYJ', 'A3PYK6HSL97EFU'],
        visualize=args.visualize,
        print_annotator=args.print_sample_from_annotator)
        
    print('\n')

