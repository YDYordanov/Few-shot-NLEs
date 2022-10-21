# Few-shot-NLEs

This is the code and dataset for the paper: [Few-Shot Out-of-Domain Transfer Learning of Natural Language Explanations](https://arxiv.org/abs/2112.06204).

You can find the small-e-WinoGrande dataset in the folder with that name.

Additional scripts and the Mechanical Turk data will be available by the end of 2022.

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
pip install transformers tensorboardx gpuinfo sentencepiece jsonlines nltk tqdm
```
4) Finally, open python and execute:
```
import nltk
nltk.download('punkt')
```


## How To Use

1) Download the [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI/tree/master/dataset), 
[WinoGrande](https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip) 
and [ComVE](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation/tree/master/ALL%20data) 
data into their corresponding Data folders.

2) Execute the data processing scripts in each Data folder to generate the train/dev/test data files.

3) To train a model run:
```
python main.py
```
