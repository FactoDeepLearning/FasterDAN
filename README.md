# Faster DAN: Multi-target Queries with Document Positional Encoding for End-to-end Handwritten Document Recognition
This repository is a public implementation of the paper: "Faster DAN: Multi-target Queries with Document Positional Encoding for End-to-end Handwritten Document Recognition".

The paper is available on [Arxiv](https://arxiv.org/abs/2301.10593).

Click to see the demo:

[![Click to see demo](https://img.youtube.com/vi/fbLGhGN7ocg/0.jpg)](https://youtu.be/fbLGhGN7ocg?list=PLW90lu1l3ue9a2I9i0CLZM_orlnxoxv3g)


Pretrained model weights are available [here](https://git.litislab.fr/dcoquenet/fasterdan) and [here](https://zenodo.org/record/7568900#.Y9Iz_hyZPbY)


Table of contents:
1. [Getting Started](#Getting-Started)
2. [Datasets](#Datasets)
3. [Training And Evaluation](#Training-and-evaluation)

## Getting Started

We used Python 3.10.4, Pytorch 1.12.0 and CUDA 10.2.

Clone the repository:

```
git clone https://github.com/FactoDeepLearning/FasterDAN.git
```

Install the dependencies in conda env:

```
conda create --name fdan
conda activate fdan
cd FasterDAN
pip install -e .
cd faster_dan
```


## Datasets
We used three datasets in the paper: RIMES 2009, READ 2016 and MAURDOR.

RIMES dataset at page level was distributed during the [evaluation compaign of 2009](https://ieeexplore.ieee.org/document/5277557).

The MAURDOR dataset was distributed during the [evaluation compaign of 2013](https://ieeexplore.ieee.org/document/6854572)

READ 2016 dataset corresponds to the one used in the [ICFHR 2016 competition on handwritten text recognition](https://ieeexplore.ieee.org/document/7814136).
It can be found [here](https://zenodo.org/record/1164045#.YiINkBvjKEA)



Raw dataset files must be placed in Datasets/raw/{dataset_name} \
where dataset name is "READ 2016", "RIMES" or "Maurdor".

## Training And Evaluation
### Step 1: Download the datasets and place the raw files in the following folder: Datasets/raw/{dataset_name}

### Step 2: Format the dataset
```
python3 Datasets/dataset_formatters/read2016_formatter.py
python3 Datasets/dataset_formatters/rimes_formatter.py
python3 Datasets/dataset_formatters/maurdor_formatter.py
```

### Step 3: Add any font you want as .ttf file in the folder Fonts

### Step 4 : Generate synthetic line dataset and pretrain on it
```
cd OCR/line_OCR/ctc/
python3 main_syn_line.py # generation
python3 main_line_ctc_syn.py # training
```
There are two lines in this script to adapt to the used dataset:
```
model.generate_syn_line_dataset("READ_2016_syn_line")
dataset_name = "READ_2016"
```

Weights and evaluation results are stored in OCR/line_OCR/ctc/outputs

### Step 6 : Training the Faster DAN / DAN
```
cd OCR/document_OCR/faster_dan/
python3 main_faster_dan.py  # faster dan
python3 main_std_dan.py  # original dan
```


Weights and evaluation results are stored in OCR/document_OCR/dan/outputs


### Remarks (for pre-training and training)
Scripts are given for the READ 2016 dataset and must be adapted for RIMES 2009 and MAURDOR (mostly dataset_name parameter, and pretraining paths)
All hyperparameters are specified and editable in the training scripts (meaning are in comments).\
Evaluation is performed just after training ending (training is stopped when the maximum elapsed time is reached or after a maximum number of epoch as specified in the training script).\
The outputs files are split into two subfolders: "checkpoints" and "results". \
"checkpoints" contains model weights for the last trained epoch and for the epoch giving the best CER on the validation set. \
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.
## Citation

```bibtex
@misc{Coquenet2022b,
  author = {Coquenet, Denis and Chatelain, Clément and Paquet, Thierry},
  title = {Faster DAN: Multi-target Queries with Document Positional Encoding for End-to-end Handwritten Document Recognition},
  year = {2023},
  doi = {10.48550/ARXIV.2301.10593},
  url = {https://arxiv.org/abs/2301.10593},
  publisher = {arXiv},
}
```

