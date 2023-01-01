"""
Running test evaluation of all reported models on their respective child tasks
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

# Create an empty gpu_synch file for GPU synchronization
synch_file_path = 'gpu_synch.txt'
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)

exper_name = 'test'


file_list = [
    'saved_models/WGft50NLEs_reproduced/25',
    'saved_models/WG+50NLEs_reproduced/22',
    'saved_models/ESNLI+SNLIftWG_reproduced/19',
    'saved_models/ESNLI+SNLI+WG_reproduced/11',
    
    'saved_models/ESNLI+WG+50NLEs_reproduced/18',
    'saved_models/ESNLI+WGft50NLEs_reproduced/16',
    'saved_models/ESNLIftWG+50NLEs_reproduced/11',
    'saved_models/ESNLIftWGft50NLEs_reproduced/22',
    
    'saved_models/ComVEft50NLEs_reproduced/26',
    'saved_models/ComVE+50NLEs_reproduced/19',
    'saved_models/ESNLI+SNLIftComVE_reproduced/16',
    'saved_models/ESNLI+SNLI+ComVE_reproduced/11',
    
    'saved_models/ESNLI+ComVE+50NLEs_reproduced/11',
    'saved_models/ESNLI+ComVEft50NLEs_reproduced/13',
    'saved_models/ESNLIftComVE+50NLEs_reproduced/1',
    'saved_models/ESNLIftComVEft50NLEs_reproduced/13'
]

for evaluate_model in file_list:
    folder_number = 1

    for b_size in [16]:
        for ep in [0]:
            for lr in [1e-4]:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                
                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                # Run command
                command = "nohup python main.py -lm=t5-base --beam_size=5 --evaluate_model={} --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=esnli -force_explain -ep={} -lr={} --scheduler=linear -bs={} -dbs=10 --grad_accum_steps=1 --gpu_synch_file={} --use_devices={} &".format(evaluate_model, exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                
                print("Executing:", command)
                os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
                folder_number += 1

