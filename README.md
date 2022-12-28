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
pip install -r requirements.txt
```
4) Finally, open python and execute:
```
import nltk
nltk.download('punkt')
```


## How To Use

1) Clone or download this repository and open the containing folder.

2) Download and prepare the data by running:
```
python download_and_process_data.py
```

3) To train a model run:
```
python main.py
```
