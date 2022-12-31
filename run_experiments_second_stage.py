"""
Run training on: e-SNLIftWG, and e-SNLIftComVE (no NLEs on child task)
Note: these models are only used as starting points of training other models.

Then run training on: e-SNLI+SNLIftWG, e-SNLI+SNLIftComVE, e-SNLIftWG+50NLEs, e-SNLIftComVE+50NLEs,
e-SNLI+WGft50NLEs, e-SNLI+ComVEft50NLEs, WGft50NLEs, ComVEft50NLEs
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

exper_name = 'ESNLIftWG_reproduced'
folder_number = 13

for b_size in [16]:
    for ep in [7]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI_3_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=ewg --ewg_train_data_path=Data/e-WG/train.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLIftComVE_reproduced'
folder_number = 23

for b_size in [16]:
    for ep in [7]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI_3_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_no_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+SNLIftWG_reproduced'
folder_number = 19

for b_size in [16]:
    for ep in [11]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI+SNLI_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=ewg --ewg_train_data_path=Data/e-WG/train.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+SNLIftComVE_reproduced'
folder_number = 16

for b_size in [16]:
    for ep in [10]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI+SNLI_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_no_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLIftWG+50NLEs_reproduced'
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
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI_3_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=ewg --ewg_train_data_path=Data/e-WG/train_50_nles.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLIftComVE+50NLEs_reproduced'
folder_number = 1

for b_size in [16]:
    for ep in [1]:
        for lr in [1e-3]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI_3_reproduced/7 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_50_nles.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+WGft50NLEs_reproduced'
folder_number = 16

for b_size in [16]:
    for ep in [10]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI+WG_reproduced/21 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=ewg --ewg_train_data_path=Data/e-WG/train_50_nles_only.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ESNLI+ComVEft50NLEs_reproduced'
folder_number = 13

for b_size in [16]:
    for ep in [5]:
        for lr in [1e-3]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ESNLI+ComVE_reproduced/11 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_50_nles_only.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'WGft50NLEs_reproduced'
folder_number = 25

for b_size in [16]:
    for ep in [21]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/WG_reproduced/21 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=1200 -task=ewg --ewg_train_data_path=Data/e-WG/train_50_nles_only.jsonl -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

exper_name = 'ComVEft50NLEs_reproduced'
folder_number = 26

for b_size in [16]:
    for ep in [13]:
        for lr in [3e-4]:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Wait for a GPU to free-up and find its idx
            gpu_idx = find_unassigned_gpu(synch_file_path)
            update_gpu_synch_file(synch_file_path, gpu_idx,
                                  is_running=True)

            # Run command
            command = "nohup python main.py -lm=t5-base --beam_size=3 --load_model=saved_models/ComVE_reproduced/18 --save_dir=saved_models/{}/{} --exper_name={} --log_interval=600 -task=comve --comve_train_data_path=Data/ComVE/train_50_nles_only.csv -ep={} -lr={} --scheduler=linear -bs={} --grad_accum_steps=4 --gpu_synch_file={} --use_devices={} &".format(exper_name, folder_number, exper_name, ep, lr, b_size, synch_file_path, gpu_idx, exper_name, folder_number)

            # Set the GPU to use
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            
            print("Executing:", command)
            os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
            folder_number += 1

