# Knowledge-aware image captioning model

## Description

This repository contains the code for training and evaluating the image captioning model with integrated encyclopedic knowledge. 
The model is trained on the [K-GeoRic](https://drive.google.com/drive/folders/1vPAfD0KIHJQvURJ9E1J538x6_BaBXQWD) image-caption dataset.

- `create_input_files.py`: contains the code that creates input files for training and evaluating the captioning model (e.g., wordmap, encoded captions of the train/validation/test split, etc.)
- `utils.py`: contains various utility functions used throughout the codebase (e.g., saving/loading data, processing/normalizing data, etc.)
- `datasets.py`: contains the code for batch loading train/validation/test data 
- `train.py`: contains the code that trains the captioning model (including the validation cycle)
- `models.py`: contains the main caption generation code - the encoder and decoder with integrated external knowledge 
- `eval.py`: contains the code for evaluating the trained captioning model on the test set
- `fact_accuracy_metric.py`: contains the code of the metric that measures the factual accuracy of the generated captions
- `compute_eval_metrics.py`: contains the code that computes the scores of the standard captioning metrics (BLEU, METEOR, ROGUE, CIDEr) for the generated captions
- `data/`: contains files with various mappings used for the factual accuracy evaluation and creating the random fact object baseline
- `img_caption_data/`: contains a JSON file with the pre-processed caption data, split into train, validation and test sets

## Training instructions

In order to train a captioning model on the K-GeoRic dataset from scratch, the following steps should be taken:
1) Create a subfolder `images/` in `img_caption_data/` and populate it with [images from the dataset](https://drive.google.com/file/d/17pVNudnd_90SSPZLfaCG2uczDvMXRJRW/view?usp=share_link). 
2) Download the [pre-trained GloVe vectors](https://nlp.stanford.edu/projects/glove/) (Common Crawl, uncased), unzip the file and put it in `data/`.
3) Put the [entity contexts and knowledge contexts](https://drive.google.com/drive/folders/1ZiCI6OfuUoPy7crkJu8fvPjqdSWveBZt) for the dataset images in `img_caption_data/`.
4) Create a folder `evalfunc/` in the root directory and put in there the [scripts for computing standard captioning metrics](https://github.com/tylin/coco-caption) (BLEU, METEOR, ROGUE, CIDEr).
5) Run `create_input_files.py` to create the input files for captioning (saved into `img_caption_data/input_dataset_files/` by default).
6) Run `train.py` to train the model.

In order to evaluate the trained model, run `eval.py`, providing the name of the checkpoint (if it is different from the default one).

## License

The code and the data in this repository are licensed for reuse under the [Creative Commons BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgments

The code infrastructure is modeled after the [PyTorch image captioning tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).

