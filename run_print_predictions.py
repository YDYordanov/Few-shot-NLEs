"""
Run some experiment (e.g. EWG hyperparam search)
"""

import argparse
from gpu_utils import *


parser = argparse.ArgumentParser(description="Transformers model")
parser.add_argument("--total_num_gpus", '-num_gpus', type=int, default=1,
                    help='Total number of GPUs in the system')
parser.add_argument("--ignore_gpu_ids", '-ignore_gpus', default=[], nargs='+',
                    type=int,
                    help='The interval-separated list of ids of the GPUs '
                         'in the system not to be used for running '
                         '(to be ignored by this script).')
args = parser.parse_args()

synch_file_path = 'gpu_synch.txt'
# Create an empty gpu_synch file for GPU synchronization
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)          

model_paths = [
    'saved_models/WGft50NLEs_reproduced/25',
    'saved_models/WG+50NLEs_reproduced/22',
    'saved_models/ESNLI+SNLIftWG_reproduced/19',
    'saved_models/ESNLI+SNLI+WG_reproduced/11',
    
    'saved_models/ESNLI+WG+50NLEs_reproduced/18',
    'saved_models/ESNLI+WGft50NLEs_reproduced/16',
    'saved_models/ESNLIftWG+50NLEs_reproduced/11',
    'saved_models/ESNLIftWGft50NLEs_reproduced/22'
]

for load_model in model_paths:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Wait for a GPU to free-up and find its idx
    gpu_idx = find_unassigned_gpu(synch_file_path)
    update_gpu_synch_file(synch_file_path, gpu_idx,
                          is_running=True)

    # Run command
    command = "nohup python main.py -lm=t5-base --evaluate_model={} --ewg_test_data_path=Data/e-WG/test_100.jsonl --save_predictions --predictions_file_name=predictions.jsonl --beam_size=5 --force_explain -task=ewg --gpu_synch_file={} --use_devices={} &".format(load_model, synch_file_path, gpu_idx)

    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    
    print("Executing:", command)
    os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)

model_paths = [    
    'saved_models/ComVEft50NLEs_reproduced/26',
    'saved_models/ComVE+50NLEs_reproduced/19',
    'saved_models/ESNLI+SNLIftComVE_reproduced/16',
    'saved_models/ESNLI+SNLI+ComVE_reproduced/11',
    
    'saved_models/ESNLI+ComVE+50NLEs_reproduced/11',
    'saved_models/ESNLI+ComVEft50NLEs_reproduced/13',
    'saved_models/ESNLIftComVE+50NLEs_reproduced/1',
    'saved_models/ESNLIftComVEft50NLEs_reproduced/13'
]

for load_model in model_paths:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Wait for a GPU to free-up and find its idx
    gpu_idx = find_unassigned_gpu(synch_file_path)
    update_gpu_synch_file(synch_file_path, gpu_idx,
                          is_running=True)

    # Run command
    command = "nohup python main.py -lm=t5-base --evaluate_model={} --comve_test_data_path=Data/ComVE/test_100.csv --save_predictions --predictions_file_name=predictions.jsonl --beam_size=5 --force_explain -task=comve --gpu_synch_file={} --use_devices={} &".format(load_model, synch_file_path, gpu_idx)

    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    
    print("Executing:", command)
    os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)

