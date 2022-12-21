# Knowledge-aware news image captioning model

## Description

This repository contains the code for training and evaluating the image captioning model with integrated encyclopedic knowledge applied to the news domain. 
The model is trained on the [NYTimes800k](https://github.com/alasdairtran/transform-and-tell) news image-caption dataset (Tran et al., 2020).

- `create_input_files.py`: contains the code that creates input files for training and evaluating the captioning model (e.g., wordmap, encoded captions of the train/validation/test split, etc.)
- `utils.py`: contains various utility functions used throughout the codebase (e.g., saving/loading data, processing/normalizing data, etc.)
- `datasets.py`: contains the code for batch loading train/validation/test data 
- `train.py`: contains the code that trains the captioning model (including the validation cycle)
- `models.py`: contains the main caption generation code - the encoder and decoder with integrated external knowledge 
- `eval.py`: contains the code for evaluating the trained captioning model on the test set
- `compute_eval_metrics.py`: contains the code that computes precision and recall of generated named entities and the scores of the standard captioning metrics (BLEU, METEOR, ROGUE, CIDEr) for the generated captions

## Training instructions

In order to train a captioning model on the NYTimes800k dataset from scratch, the following steps should be taken:
1) Create a folder `img_caption_data/images/` in the root directory and populate it with images from the dataset. 
2) Create a folder `data/` in the root directory and put in there the unzipped [pre-trained GloVe vectors](https://nlp.stanford.edu/projects/glove/) (Common Crawl, uncased).
3) Put the entity contexts and knowledge contexts for the dataset images in `img_caption_data/`.
4) Put the file with the pre-processed caption data, split into train, validation and test sets in `img_caption_data/`.
5) Create a folder `evalfunc/` in the root directory and put in there the [scripts for computing standard captioning metrics](https://github.com/tylin/coco-caption) (BLEU, METEOR, ROGUE, CIDEr).
6) Run `create_input_files.py` to create the input files for captioning (saved into `img_caption_data/input_dataset_files/` by default).
7) Run `train.py` to train the model.

In order to evaluate the trained model, run `eval.py`, providing the name of the checkpoint (if it is different from the default one).

NB! All the data required for training and evaluating the model (including what was mentioned above: the images, entity and knowledge contexts, caption split data) is provided on demand (due to the dataset size). The request can be sent to `s.nikiforova@uu.nl`.

## License

The code and the data in this repository are licensed for reuse under the [Creative Commons BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgments

- The code infrastructure is modeled after the [PyTorch image captioning tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
- The NYTimes800k dataset was presented in (Tran et al., 2020) and distributed by the [dataset creators](https://github.com/alasdairtran/transform-and-tell) for research purposes.

Tran, A., Mathews, A., and Xie, L. (2020). Transform and Tell: Entity-aware news image captioning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13035–13045.
