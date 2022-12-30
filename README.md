# Few-shot-NLEs

This is the code and dataset for the paper: [Few-Shot Out-of-Domain Transfer Learning of Natural Language
Explanations in a Label-Abundant Setup](https://arxiv.org/abs/2112.06204).

You can find the small-e-WinoGrande dataset in the folder with that name.

You can find the raw Mechanical Turk data in the Results folder.

Additional scripts will be available by the end of 2022.

## Installation instructions

These instructions assume [Anaconda](https://www.anaconda.com).
1) Create a new Anaconda environment and activate it:
```
conda create -n nles python numpy
conda activate nles
```
2) Install pytorch via its [official installation instructions for Anaconda](https://pytorch.org/get-started/locally/).
3) Install via pip:
```
pip install -r requirements.txt
```
4) Finally, open python and execute:
```
import nltk
nltk.download('punkt')
```


## How to use

1) Clone or download this repository and open the containing folder.

2) Download and prepare the data by running:
```
python download_and_process_data.py
```

3) To train a model run:
```
python main.py
```

An example of multi-task learning on e-SNLI and WinoGrande with 50 NLEs:
```
python main.py --esnli_train_data_path=Data/e-SNLI/esnli_train.csv --ewg_train_data_path=Data/e-WG/train_50_nles.jsonl 
--lm_name=t5-base --beam_size=3 
--save_dir=saved_models/Experiment_1/Run_1 --exper_name=Experiment1 --log_interval=5000 -task=esnli+ewg 
--num_epochs=2 --lr=1e-3 --scheduler=linear --train_b_size=16 --grad_accum_steps=2
```
Tip: use ```--grad_accum_steps``` to specify gradient accumulation (value between 1 and ```train_b_size```). It helps with fitting large models on a small GPU.

## How to reproduce the results

The command below reproduces the human NLE evaluation results in Table 5 of the Appendix. 
The script processes the raw Mechanical Turk data located in the Results folder.    

```
python create_mturk_results_table.py
```
