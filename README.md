# Few-shot-NLEs

This is the code and dataset for the paper: [Few-Shot Out-of-Domain Transfer Learning of Natural Language
Explanations in a Label-Abundant Setup](https://arxiv.org/abs/2112.06204).

You can find the small-e-WinoGrande dataset in the folder with that name.

You can find the raw Mechanical Turk data in the Results folder.


## Installation instructions

These instructions assume Anaconda ([link](https://www.anaconda.com)).

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

4) Clone or download this repository and open the containing folder.

5) Open Python and execute:
```
import nltk
nltk.download('punkt')
```

6) Download and prepare the data by running:
```
python download_and_process_data.py
```


## How to use

To train a model run:
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

### How to reproduce the tables

The command below reproduces the human NLE evaluation results in Table 5 of the Appendix, and, in particular, 
the "NLE score" columns in Table 3 of the main paper. 
The script processes the raw Mechanical Turk data located in the Results folder.

```
python create_mturk_results_table.py
```

The command below reproduces the ComVE automatic NLE evaluation results in Table 3 of the main paper. 
The script processes the NLE generations from all ComVE models (located in the "saved_models" folder).

```
python automatic_evaluation.py
```

### How to train the models

The commands below execute the training of all models in Table 3 of the results, in three stages. Note: please wait for each script to finish running 
before starting the next one due to model inter-dependence.

```
python run_experiments_first_stage.py -num_gpus=1
python run_experiments_second_stage.py -num_gpus=1
python run_experiments_third_stage.py -num_gpus=1
```
Tip: use ```-num_gpus``` to specify how many GPUs to use for training.

The commands below generate the test result accuracies on the child datasets and format them into a table.
```
python run_test_evaluate_best_models.py -num_gpus=1
python create_table_test_results.py
```

The command below generates NLEs from all models on 100 test instances. The generated NLEs are saved as 
"predictions.jsonl" files located in the corresponding model directories.
```
python run_print_predictions.py -num_gpus=1
```
