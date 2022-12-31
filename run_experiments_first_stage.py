"""
Run training on: WG, ComVE, e-SNLI, e-SNLI+SNLI, e-SNLI+WG, and e-SNLI+ComVE (no NLEs on child tasks) 
Note: these models are only used as starting points of training other models.

Then run training on: WG+50NLEs, ComVE+50NLEs, e-SNLI+SNLI+WG, e-SNLI+SNLI+ComVE, e-SNLI+WG+50NLEs, and e-SNLI+ComVE+50NLEs
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

exper_name = 'WG_reproduced'
folder_number = 21

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1000 -task=ewg --ewg_train_data_path=Data/e-WG/train.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ComVE_reproduced'
folder_number = 18

for b_size in [16]:
    for ep in [5]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_no_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI_reproduced'
folder_number = 7

for b_size in [16]:
    for ep in [3]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=1 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+SNLI_reproduced'
folder_number = 7

for b_size in [16]:
    for ep in [3]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --esnli_train_data_path=Data/e-SNLI/esnli+snli_train.csv --beam_size=1 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+WG_reproduced'
folder_number = 21

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+ewg --ewg_train_data_path=Data/e-WG/train.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+ComVE_reproduced'
folder_number = 11

for b_size in [16]:
    for ep in [3]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+comve --comve_train_data_path=Data/ComVE/train_no_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1
            
exper_name = 'WG+50NLEs_reproduced'
folder_number = 22

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1000 -task=ewg --ewg_train_data_path=Data/e-WG/train_50_nles.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ComVE+50NLEs_reproduced'
folder_number = 19

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_50_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+SNLI+WG_reproduced'
folder_number = 11

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --esnli_train_data_path=Data/e-SNLI/esnli+snli_train.csv --beam_size=1 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+ewg -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+SNLI+ComVE_reproduced'
folder_number = 11

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --esnli_train_data_path=Data/e-SNLI/esnli+snli_train.csv --beam_size=1 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+comve -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1
            
exper_name = 'ESNLI+WG+50NLEs_reproduced'
folder_number = 18

for b_size in [16]:
    for ep in [3]:
        for lr in [3e-5]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+ewg --ewg_train_data_path=Data/e-WG/train_50_nles.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+ComVE+50NLEs_reproduced'
folder_number = 11

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=5000 -task=esnli+comve --comve_train_data_path=Data/ComVE/train_50_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

