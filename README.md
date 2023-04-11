# Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning

This repository contains the code to replicate the experiments in the paper *Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning* (forthcoming), as well as an improved implementation for the evaluated models (CLSWGAN, TFVAEGAN, and FREE).

## Abstract

> One of the main challenges of deep learning is the need for large amounts of data to effectively train models, and methods such as zero-shot learning (ZSL) have been developed to address this issue. ZSL consists of training models on a set of "seen" classes and evaluating them on a set of "unseen" classes. While ZSL has demonstrated promising results, particularly with generative methods, its generalizability to real-world scenarios remains uncertain.
> 
> In this paper, we first introduce the concepts of generalizability and robustness for attribute-based zero-shot learning, then we conduct a series of experiments to evaluate these properties in ZSL models. The aim of this evaluation is to lay the foundation for further research on the generalizability and robustness of ZSL models, and to apply these findings to real-world applications.
> 
> Our contribution is twofold. First, we conduct a series of experiments to test the generalizability of ZSL models by evaluating their performance on different training conditions, or "splits". Then, we assess the robustness by evaluating models on splits with specific properties, with the aim of stress testing the models themselves. We evaluate the accuracy of state-of-the-art models on coarse- and fine-grained benchmark datasets, and we demonstrate a significant margin for improvement in generalizability and robustness.

## Code

This repository consists of four main parts:

- `run_clswgan.py`: PyTorch implementation of the CLSWGAN model from the paper [Feature Generating Networks for Zero-Shot Learning](https://arxiv.org/abs/1712.00981).
- `run_tfvaegan.py`: PyTorch implementation of the TFVAEGAN model from the paper [Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670477.pdf).
- `run_free.py`: PyTorch implementation of the FREE model from the paper [FREE: Feature Refinement for Generalized Zero-Shot Learning](https://arxiv.org/abs/2107.13807).
- `run_splitter.py`: Implementation of the *splitter* proposed in the paper Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning (forthcoming).

The code for the three models has been adapted from the original repositories ([CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/), [TFVAEGAN](https://github.com/akshitac8/tfvaegan), [FREE](https://github.com/shiming-chen/FREE)). The code provided here is a simplified version of the original code which has been heavily refactored, documented, and optimized for clarity and extensibility. The code for the splitter is original.

The splitter uses particular methods to generate new splits of seen/unseen classes and attributes in the original dataset. Three splitting methods are currently implemented, although more will be added in the future: Greedy Class Split (GCS), Clustered Class Split (CCS), and Minimal Attribute Split (MAS).

The three models are trained and tested on the generated splits to evaluate their generalizability and robustness. The results are reported in the forthcoming paper.

## Installation

The code has been tested with Python 3.8.10 and PyTorch 1.13.1+cu116. To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

Before running the code, you need to download the datasets in the `data` folder. 4 datasets are currently supported: AWA2, CUB, SUN, and FLO. The datasets are available [here](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) (if the link doesn't work, copy it in a new tab instead of clicking on it). For each dataset, you only require 2 of the downloaded files: `res101.mat` with the features, and `att_splits.mat` with the attributes. Make sure you have these files in the appropriate folder for each dataset, e.g. `data/awa/res101.mat` and `data/awa/att_splits.mat` for the AWA2 dataset.

You can either run the scripts directly with your own arguments (defined in `args.py`), e.g.:

```bash
python run_clswgan.py --dataset AWA2 --split gcs --n_epochs 20
```

or you can use the provided scripts to run the experiments from the paper:

```bash
python scripts/run_clswgan_awa_gzsl.py
```

This will train and evaluate the model on the selected dataset and split.

To generate the splits, run `run_splitter.py` with the desired dataset and splitting method, e.g.:

```bash
python run_splitter.py --dataset AWA2 --split gcs
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
