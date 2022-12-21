from collections import Counter
import h5py
from imageio import imread
import json
import numpy as np
import os
import pickle
from skimage.transform import resize
from tqdm import tqdm

import utils as ut


class InputConstructor:
    def __init__(
        self,
        splits_captions_json_path,
        entity_context_path,
        image_dir,
        wordmap_file_path,
        update_wordmap,
        min_word_freq,
        output_dir,
        to_base_name,
        max_len,
        overwrite,
    ):
        """
        Creates input files for training, validation and test data.

        :param splits_captions_json_path: path to the JSON file with the caption data split into train-val-test sets
        :param entity_context_path: path to the pickle file with the entity context for each image
            - a dict of the format {url: df with the entity context}
        :param image_dir: path to the folder with the images
        :param wordmap_file_path: path to the JSON file with the wordmap
            - if None, construct the wordmap from the input captions
        :param update_wordmap: whether or not to update the precompiled wordmap with the input captions' tokens
            - only relevant if `wordmap_file_path` is not None
        :param min_word_freq: words occuring less frequently than this threshold are replaced with the dummy <unk> token
        :param output_dir: path to the folder to save the files in
        :param to_base_name: a suffix to append to the base file name
        :param max_len: captions longer than this length are skipped
        :param overwrite: whether to overwrite the files if they already exist
        """
        #
        # Create a base/root name for all output files.
        self.base_filename = "geo_aware" + to_base_name
        #
        # Read the JSON with the train-val-test caption sets.
        with open(splits_captions_json_path, "r") as j:
            self.data = json.load(j)
        # Read the entity contexts for the images.
        with open(entity_context_path, "rb") as f:
            self.entity_contexts = pickle.load(f)
        self.entity_context_size = 300
        #
        self.img_dim_size = 256
        self.image_dir = image_dir
        self.wordmap_file_path  = wordmap_file_path
        self.update_wordmap = update_wordmap
        self.min_word_freq = min_word_freq
        self.output_dir = output_dir
        self.max_len = max_len
        self.overwrite = overwrite

    def run(self):
        # Create the wordmap.
        word_map = self._construct_word_map()
        #
        # Collect the data.
        split_types = ["TRAIN", "VAL", "TEST"]
        data_types = [
            "image_path", "caption", "caption_mask", "caption_length", 
            "entity_features", "entity_names"
        ]
        caption_data = {}
        for split in split_types:
            caption_data[split] = {}
            for data_type in data_types:
                caption_data[split][data_type] = []
        #
        outputs = [self.get_data_for_image(img, word_map) for img in tqdm(self.data["images"]) if len(img["tokens"]) <= self.max_len]
        for output in outputs:
            split = output["split"]
            for data_type in data_types:
                caption_data[split][data_type].append(output[data_type])        
        #
        # Save the data.
        for split in split_types:
            hdf5_filename = os.path.join(
                self.output_dir,
                split + "_IMAGES_" + self.base_filename + ".hdf5",
            )
            if os.path.exists(hdf5_filename) and self.overwrite:
                # Delete the file if it already exists.
                os.remove(hdf5_filename)
            with h5py.File(hdf5_filename, "a") as h:
                # Create a dataset inside HDF5 file to store the images.
                images = h.create_dataset(
                    "images", (len(caption_data[split]["image_path"]), 3, self.img_dim_size, self.img_dim_size), dtype="float16"
                )
                #
                print(
                    "\nReading %s images and captions, storing to file...\n"
                    % split
                )
                #
                for i, img_path in enumerate(tqdm(caption_data[split]["image_path"])):
                    # Save the image to HDF5 file.
                    processed_image = self._prepare_image(img_path)
                    images[i] = processed_image
                #
                # Sanity check.
                assert images.shape[0] == len(caption_data[split]["caption"])
                del images
                #
                with open(os.path.join(self.output_dir, split + "_CAPTIONS_" + self.base_filename + ".json"), "w") as fh:
                    json.dump(caption_data[split]["caption"], fh)
                
                with open(os.path.join(self.output_dir, split + "_CAPLENS_" + self.base_filename + ".json"), "w") as fh:
                    json.dump(caption_data[split]["caption_length"], fh)
                
                with open(os.path.join(self.output_dir, split + "_CAPMASKS_" + self.base_filename + ".json"), "w") as fh:
                    json.dump(caption_data[split]["caption_mask"], fh)
                
                with open(os.path.join(self.output_dir, split + "_ENT_FEATURES_" + self.base_filename + ".pkl"), "wb") as fh:
                    pickle.dump(caption_data[split]["entity_features"], fh)
                
                with open(os.path.join(self.output_dir, split + "_ENT_NAMES_" + self.base_filename + ".pkl"), "wb") as fh:
                    pickle.dump(caption_data[split]["entity_names"], fh)
                
                            
    def get_data_for_image(self, img, word_map):
        """
        Get captioning data for a single image.
        
        :param img: the data for the image from the train-val-test split file
            - this data includes the caption tokens, mask, url, id and split label
        :param word_map: the wordmap (vocabulary)
        :return: the captioning data for the image, in a form of a dict with the following fields
            - "split": the image split label ("TRAIN", "VAL", "TEST")
            - "image_path": path to the image location
            - "caption": the caption encoded as a series of token indices
            - "caption_mask": the mask indicating which of the caption tokens are vocab words and 
              which are entity names
            - "caption_length": the length of the encoded caption (including the <start> and <end> dummy tokens)
            - "entity_features": the features of the entities in the entity context
            - "entity_names": the names of the entities in the entity context
        """
        # Process the entity context for the image.
        image_url = img["url"]
        entity_context_for_image = self.entity_contexts[image_url]
        (
            entity_features,
            entity_names,
        ) = ut.prepare_context(
            entity_context_for_image,
            name_col="name",
            feature_cols = ["distance", "azimuth", "size", "type_index"],
            random_value_range = [(0.0, 1.0), (-179.0, 179.0), (0.0, 0.1), (0, 500)],
            context_size=self.entity_context_size,
            unk_dummy="<unk_ent>"
        )
        # Process the caption for the image.
        (
            encoded_caption, 
            encoded_caption_mask, 
            encoded_caption_length,
        ) = self.process_caption(img, word_map, entity_names)
        # Get the path to the image itself.
        path = os.path.join(self.image_dir, img["id"])
        if not os.path.exists(path):
            path = path + ".jpg"
        #
        out_dict = {
            "split": img["split"].upper(), 
            "image_path": path, 
            "caption": encoded_caption,
            "caption_mask": encoded_caption_mask,
            "caption_length": encoded_caption_length,
            "entity_features": entity_features , 
            "entity_names": entity_names, 
        }
        return out_dict
    
    def process_caption(self, img, word_map, entity_names):
        """
        Encode a caption turning its tokens into numerical indices.
        
        :param img: the data for the image from the train-val-test split file
            - this data includes the caption tokens, mask, url, id and split label
        :param word_map: the wordmap (vocabulary)
        :param entity_names: the names of the entities in the entity context
        :return: 
            - the caption encoded as a series of token indices
            - the mask indicating which of the caption tokens are vocab words and 
              which are entity names
            - the length of the encoded caption (including the <start> and <end> dummy tokens)

        """
        caption_tokens = img["tokens"]
        caption_mask = img["mask"]
        #
        # Encode the caption.
        ent_int_name_to_index = ut.compile_entity_name_index(entity_names)
        # Start with the <start> token, which is a regular vocab token (mask label "0").
        encoded_caption = [word_map["<start>"]]
        encoded_caption_mask = [0]
        for token_num in range(len(caption_tokens)):
            # If it's a regular vocab word, get the word index from the wordmap.
            if caption_mask[token_num] == 0:
                encoded_caption_mask.append(0)
                encoded_caption.append(
                    word_map.get(caption_tokens[token_num], word_map["<unk>"])
                )
            #
            # If it's an entity, get its index from the entity name dict (compiled from the entity context).
            elif caption_mask[token_num] == 1:
                encoded_caption_mask.append(1)
                # Encode the entity name into a series of integers.
                int_name_list = tuple(ut.str_to_int(caption_tokens[token_num]))
                enc_ent_name = ""
                if int_name_list in ent_int_name_to_index:
                    enc_ent_name = ent_int_name_to_index[int_name_list]
                else:
                    # Look for the name with fuzzy matching.
                    name_to_str = "^".join(
                        [
                            str(x)
                            for x in list(int_name_list)
                            if x != ut.DUMMY_CHAR_ENCODING
                        ]
                    )
                    longest_match_key = ""
                    longest_match_len = 0
                    for key in ent_int_name_to_index:
                        key_to_str = "^".join(
                            [
                                str(x)
                                for x in list(key)
                                if x != ut.DUMMY_CHAR_ENCODING
                            ]
                        )
                        if key_to_str in name_to_str or name_to_str in key_to_str:
                            if len(key_to_str) > longest_match_len:
                                longest_match_len = len(key_to_str)
                                longest_match_key = key
                    if longest_match_key != "":
                        enc_ent_name = ent_int_name_to_index[longest_match_key]
                if enc_ent_name == "":
                    # If the name is absent from the entity context, encode as "<unk_ent>".
                    unk_ent_encoding = tuple(ut.str_to_int("<unk_ent>"))
                    enc_ent_name = ent_int_name_to_index[unk_ent_encoding]
                    encoded_caption.append(len(word_map) + enc_ent_name)
                else:
                    encoded_caption.append(len(word_map) + enc_ent_name)
        # Add the <end> token and <pad> tokens to pad to the maximum length.
        encoded_caption = (
            encoded_caption
            + [word_map["<end>"]]
            + [word_map["<pad>"]] * (self.max_len - len(caption_tokens))
        )
        # <end> and <pad> tokens are considered to be regular vocab tokens (mask label "0").
        encoded_caption_mask = (
            encoded_caption_mask + [0] + [0] * (self.max_len - len(caption_tokens))
        )
        # Define the caption length (incl. <start> and <end> tokens).
        encoded_caption_length = len(caption_tokens) + 2
        return encoded_caption, encoded_caption_mask, encoded_caption_length

    def _construct_word_map(self):
        """
        Build the wordmap for encoding the captions.
        
        The wordmap is either loaded from a file or compiled from scratch from the words
        in the captions with frequency higher than the indicated minimum value.
        If the wordmap is loaded from a file, it can still be updated with the words in the captions, if specified.
        
        :return: the wordmap for encoding the captions
        """
        if self.wordmap_file_path:
            # Load wordmap from a file.
            with open(self.wordmap_file_path, "r") as j:
                word_map = json.load(j)
            if not self.update_wordmap:
                # Save to a file.
                with open(os.path.join(self.output_dir, "WORDMAP_" + self.base_filename + ".json"), "w") as j:
                    json.dump(word_map, j)
                return word_map
        # Get the tokens for the wordmap.
        word_freq_counter = Counter()
        for img in self.data["images"]:
            if img["split"] == "train":
                # Update word frequency with the vocab words.
                vocab_tokens = [x for ix, x in enumerate(img["tokens"]) if img["mask"][ix] == 0]
                word_freq_counter.update(vocab_tokens)
        words = [
            w
            for w in word_freq_counter.keys()
            if word_freq_counter[w] > self.min_word_freq
        ]
        #
        if self.wordmap_file_path and self.update_wordmap:
            # Update the wordmap with the words from captions.
            del word_map["<unk>"]
            del word_map["<start>"]
            del word_map["<end>"]
            del word_map["<pad>"]
            words_to_add = [w for w in word_map if w not in words]
            words.extend(words_to_add)	 
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map["<unk>"] = len(word_map) + 1
        word_map["<start>"] = len(word_map) + 1
        word_map["<end>"] = len(word_map) + 1
        word_map["<pad>"] = 0
        # Save to a file.
        with open(os.path.join(self.output_dir, "WORDMAP_" + self.base_filename + ".json"), "w") as j:
            json.dump(word_map, j)
        return word_map
    
    def _prepare_image(self, image_path):
        """
        Process the image: resize, reshape to the shape suitable for the captioning encoder.
        
        :param image_path: the location of the image
        :return: the processed image
        """
        processed_img = imread(image_path)
        if len(processed_img.shape) == 2:
            processed_img = processed_img[:, :, np.newaxis]
            processed_img = np.concatenate(
                [processed_img, processed_img, processed_img], axis=2
            )
        processed_img = resize(processed_img, (self.img_dim_size, self.img_dim_size))
        processed_img = processed_img.transpose(2, 0, 1)
        assert processed_img.shape == (3, self.img_dim_size, self.img_dim_size)
        assert np.max(processed_img) <= self.img_dim_size-1
        return processed_img
        

if __name__ == "__main__":
    #
    input_constructor = InputConstructor(
        splits_captions_json_path="img_caption_data/captions_split.json",
        entity_context_path="img_caption_data/entity_context.pkl",
        image_dir="img_caption_data/images/",
        wordmap_file_path=None,
        update_wordmap=False,
        min_word_freq=0,  #  --> keep all the words
        output_dir="img_caption_data/input_dataset_files/",
        to_base_name="_georic2",
        max_len=30,
        overwrite=True,
    )
    #
    input_constructor.run()
