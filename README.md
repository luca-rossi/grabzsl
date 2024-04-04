# Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning

This repository contains the code to replicate the experiments in the paper *Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning*, as well as an improved implementation for the evaluated models (CLSWGAN, TFVAEGAN, and FREE).

## Citation

If you find this work useful, please consider citing the paper:

```bibtex
@article{rossi2024generalizability
  title = {Generalizability and robustness evaluation of attribute-based zero-shot learning},
  author = {Luca Rossi and Maria Chiara Fiorentino and Adriano Mancini and Marina Paolanti and Riccardo Rosati and Primo Zingaretti},
  journal = {Neural Networks},
  pages = {106278},
  year = {2024},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2024.106278},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608024002028}
}
```

## Abstract

> In the field of deep learning, large quantities of data are typically required to effectively train models. This challenge has given rise to techniques like zero-shot learning (ZSL), which trains models on a set of "seen" classes and evaluates them on a set of "unseen" classes. Although ZSL has shown considerable potential, particularly with the employment of generative methods, its generalizability to real-world scenarios remains uncertain.
> 
> The hypothesis of this work is that the performance of ZSL models is systematically influenced by the chosen "splits"; in particular, the statistical properties of the classes and attributes used in training. In this paper, we test this hypothesis by introducing the concepts of generalizability and robustness in attribute-based ZSL and carry out a variety of experiments to stress-test ZSL models against different splits. Our aim is to lay the groundwork for future research on ZSL models' generalizability, robustness, and practical applications.
> 
> We evaluate the accuracy of state-of-the-art models on benchmark datasets and identify consistent trends in generalizability and robustness. We analyze how these properties vary based on the dataset type, differentiating between coarse- and fine-grained datasets, and our findings indicate significant room for improvement in both generalizability and robustness. Furthermore, our results demonstrate the effectiveness of dimensionality reduction techniques in improving the performance of state-of-the-art models in fine-grained datasets.

## Code

This repository consists of four main parts:

- `run_clswgan.py`: PyTorch implementation of the CLSWGAN model from the paper [Feature Generating Networks for Zero-Shot Learning](https://arxiv.org/abs/1712.00981).
- `run_tfvaegan.py`: PyTorch implementation of the TFVAEGAN model from the paper [Latent Embedding Feedback and Discriminative Features for Zero-Shot Classification](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670477.pdf).
- `run_free.py`: PyTorch implementation of the FREE model from the paper [FREE: Feature Refinement for Generalized Zero-Shot Learning](https://arxiv.org/abs/2107.13807).
- `run_splitter.py`: Implementation of the *splitter* proposed in the paper [Generalizability and Robustness Evaluation of Attribute-Based Zero-Shot Learning](https://www.sciencedirect.com/science/article/pii/S0893608024002028)

The code for the three models has been adapted from the original repositories ([CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/), [TFVAEGAN](https://github.com/akshitac8/tfvaegan), [FREE](https://github.com/shiming-chen/FREE)). The code provided here is a simplified version of the original code which has been heavily refactored, documented, and optimized for clarity and extensibility. The code for the splitter is original.

The splitter uses particular methods to generate new splits of seen/unseen classes and attributes in the original dataset. Four splitting methods are currently implemented, although more will be added in the future: Greedy Class Split (GCS), Clustered Class Split (CCS), Minimal Attribute Split (MAS), and PCA Attribute Split (PAS).

The three models are trained and tested on the generated splits to evaluate their generalizability and robustness. The results are reported in the paper.

## Installation

The code has been tested with Python 3.8.10 and PyTorch 1.13.1+cu116. To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

Before running the code, you need to download the datasets in the `data` folder. 4 datasets are currently supported: AWA2, CUB, SUN, and FLO. The datasets are available [here](https://drive.google.com/drive/folders/16Xk1eFSWjQTtuQivTogMmvL3P6F_084u). For each dataset, you only require 2 of the downloaded files: `res101.mat` with the features, and `att_splits.mat` with the attributes. Make sure you have these files in the appropriate folder for each dataset, e.g. `data/awa/res101.mat` and `data/awa/att_splits.mat` for the AWA2 dataset.

You can run the scripts directly with the arguments defined in `args.py`, e.g.:

```bash
python run_clswgan.py --dataset AWA2 --split gcs --n_epochs 20
```

This will train and evaluate the model on the selected dataset and split.

To generate the splits, run `run_splitter.py` with the desired dataset and splitting method, e.g.:

```bash
python run_splitter.py --dataset AWA2 --split gcs
```

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
