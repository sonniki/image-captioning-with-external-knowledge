import h5py
import json
import os
import pickle
import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):

    def __init__(self, data_dir, data_name, split, transform=None):
        """
        :param data_dir: path to the folder where data files are stored
        :param data_name: the base name of all the files
        :param split: split type ("TRAIN", "VAL", "TEST")
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {"TRAIN", "VAL", "TEST"}
        #
        # Open hdf5 file where the images are stored.
        self.h = h5py.File(os.path.join(data_dir, self.split + "_IMAGES_" + data_name + ".hdf5"), "r",)
        self.imgs = self.h["images"]
        # Load the encoded captions.
        with open(os.path.join(data_dir, self.split + "_CAPTIONS_" + data_name + ".json"), "r") as j:
            self.captions = json.load(j)
        # Load the caption lengths.
        with open(os.path.join(data_dir, self.split + "_CAPLENS_" + data_name + ".json"), "r") as j:
            self.caplens = json.load(j)
        # Load the caption masks.
        with open(os.path.join(data_dir, self.split + "_CAPMASKS_" + data_name + ".json"), "r") as j:
            self.capmasks = json.load(j)
        # Load the entity features and names.
        with open(os.path.join(data_dir, self.split + "_ENT_FEATURES_" + data_name + ".pkl"), "rb") as j:
            self.entity_features = pickle.load(j)
        with open(os.path.join(data_dir, self.split + "_ENT_NAMES_" + data_name + ".pkl"), "rb") as j:
            self.entity_names = pickle.load(j)
        # Load facts and fact names.
        with open(os.path.join(data_dir, self.split + "_FACTS_" + data_name + ".pkl"), "rb") as j:
            self.facts = pickle.load(j)
        with open(os.path.join(data_dir, self.split + "_FACT_NAMES_" + data_name + ".pkl"), "rb") as j:
            self.fact_names = pickle.load(j)
        # PyTorch transformation pipeline for the images (normalizing, etc.).
        self.transform = transform
        # Total number of data points.
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.0)
        if self.transform is not None:
            img = self.transform(img)
        #
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        capmask = torch.LongTensor(self.capmasks[i])
        #
        img_entity_features = torch.Tensor([x for x in self.entity_features[i]])
        img_entity_names = torch.LongTensor([x for x in self.entity_names[i]])
        #
        img_facts = torch.LongTensor([x for x in self.facts[i]])
        img_fact_names = torch.LongTensor([x for x in self.fact_names[i]])
        return img, caption, caplen, capmask, img_entity_features, img_entity_names, img_facts, img_fact_names

    def __len__(self):
        return self.dataset_size