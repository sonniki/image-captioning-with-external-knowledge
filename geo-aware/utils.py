import numpy as np
import random
import re
import torch
from tqdm import tqdm


def save_checkpoint(
    data_name,
    epoch,
    epochs_since_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    loss,
    is_best,
):
    """
    Save the model checkpoint.

    :param data_name: base name of the files
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since the last improvement
    :param encoder: encoder model to save
    :param decoder: decoder model to save
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning is enabled
    :param decoder_optimizer: optimizer to update decoder's weights
    :param loss: loss for the current epoch
    :param is_best: whether this is the best checkpoint so far according to the loss
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "loss": loss,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    # Store every 2nd checkpoint.
    if epoch % 2 == 0:
        filename = "checkpoint_" + str(epoch) + "_" + data_name + ".pth.tar"
    else:
        filename = "checkpoint_" + data_name + ".pth.tar"
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint.
    if is_best:
        torch.save(state, "BEST_" + filename)

class AverageMeter(object):
    """
    Keep track of the most recent, average, sum and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
 
####################################################
### LEARNING #######################################
####################################################

def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrink learning rate by the specified factor.

    :param optimizer: optimizer the learning rate of which must be shrunk
    :param shrink_factor: factor in the interval (0, 1) to multiply the learning rate by
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


####################################################
### EMBEDDINGS ####################################
####################################################

def load_embeddings(emb_file, word_map):
    """
    Create an embedding matrix for the wordmap.

    :param emb_file: file with pre-trained embeddings (stored in the GloVe format)
    :param word_map: wordmap in the format {word: idx}
    :return: embeddings in the same order as the words in the wordmap
    """
    # Determine the target embedding dimension.
    with open(emb_file, "r") as f:
        emb_dim = len(f.readline().split(" ")) - 1
    #
    vocab = set(word_map.keys())
    # Create and initialize a tensor to hold the embeddings.
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)
    # Read the embedding file.
    print("\nLoading embeddings...")
    for line in tqdm(open(emb_file, "r")):
        line = line.split()
        emb_word = line[0]
        # If the word is not in the wordmap, skip.
        if emb_word not in vocab:
            continue
        # Construct the embedding.
        embedding = list(
            map(
                lambda t: float(t),
                filter(lambda n: n and not n.isspace(), line[1:]),
            )
        )
        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
    return embeddings

def init_embedding(embeddings):
    """
    Initialize the embedding matrix with values from the uniform distribution.

    :param embeddings: the embedding matrix
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


##################################################################
### PREPARING ENTITIES FOR CAPTIONING ############################
##################################################################

DUMMY_CHAR_ENCODING = 124  # 124 is ord('|'), which does not occur in entity names and is removed from captions.

def str_to_int(input_str):
    """
    Encode the string text into a series of integers.
    Each character is encoded as an integer.
    
    :param input_str: the string to encode
    :return: the integer encoding of the input string
    """
    max_str_length = 50
    str_characters_list = []
    for char in input_str:
        # Encode each character with an integer.
        str_characters_list.append(ord(char))
    if len(str_characters_list) > max_str_length:
        # Cut down the string encoding.
        str_characters_list = str_characters_list[:max_str_length]
    elif len(str_characters_list) < max_str_length:
        # Pad the encoding with the dummy char encoding.
        diff = max_str_length - len(str_characters_list)
        for _ in range(diff):
            str_characters_list.append(DUMMY_CHAR_ENCODING)
    return str_characters_list

def int_to_str(str_characters_list, len_str):
    """
    Decode a series of integers back into a text string.
    
    :param str_characters_list: a list of integers that encode a string
    :param len_str: the expected length of the output string
    :return: the decoded string
    """
    output_str = ""
    for char in str_characters_list:
        if len(output_str) == len_str:
            # Decoding is finished.
            return output_str
        # Decode each integer into a character.
        output_str += chr(char)
    return output_str

def prepare_context(context_for_image, name_col, feature_cols, random_value_range, context_size, unk_dummy):
    """
    Prepare entity context to use during captioning.
    
    :param context_for_image: the entity context for the image as a pd.DataFrame
    :param name_col: the name of the column with the entity name
    :param feature_cols: the names of the columns with the entity features
    :param random_value_range: the ranges of values to randomly select from for each of the entity features
        - assigned to the dummies and used for padding
    :param context_size: the number of elements to keep in the context
        - since the number of elements has to be the same for every image, if the original number 
          exceeds `context_size`, the extra elements are removed; if the original number is lower
          than `context_size`, the context is padded with dummies
    :param unk_dummy: the name to give the dummy token (used, for example, for padding)
    :return: the entity context in the format suitable for captioning, including
        - features of all the context elements
        - names of all the context elements
    """
    assert len(feature_cols) == len(random_value_range)
    #
    features = []
    names = []
    for row_i in range(len(context_for_image)):
        row = context_for_image.iloc[row_i]
        # Encode the name.
        name_processed = normalize_name(row[name_col])
        if not len(name_processed):
            continue
        name_characters_list = str_to_int(name_processed)
        # Store the features.
        row_data = [row_i]
        for col in feature_cols:
            row_data.append(row[col])
        features.append(row_data)
        names.append(
            [row_i, len(name_processed)] + name_characters_list
        )
    # Ensure that the size of the context is always the same.
    pad_size = context_size - len(features)
    if pad_size < 0:
        # Cut off at the limit.
        features = features[: context_size]
        names = names[: context_size]
    else:
        for _ in range(pad_size):
            # Pad with a dummy.
            cur_list_len = len(features)
            # Assign random feature values.
            random_row_data = [cur_list_len]
            for val_range in random_value_range:
                if type(val_range[0]) == int:
                    random_val = random.randint(val_range[0], val_range[1])
                else:
                    random_val = random.uniform(val_range[0], val_range[1])
                random_row_data.append(random_val)
            features.append(random_row_data)
            names.append(
                [cur_list_len, len(unk_dummy)] + str_to_int(unk_dummy)
            )
    # Add a dummy for the case when a caption contains an entity that is not in the context.
    cur_list_len = len(features)
    # Assign random feature values.
    random_row_data = [cur_list_len]
    for val_range in random_value_range:
        if type(val_range[0]) == int:
            random_val = random.randint(val_range[0], val_range[1])
        else:
            random_val = random.uniform(val_range[0], val_range[1])
        random_row_data.append(random_val)
    features.append(random_row_data)
    names.append(
        [cur_list_len, len(unk_dummy)] + str_to_int(unk_dummy)
    )
    # Sanity check.
    assert len(features) == context_size + 1 == len(names)
    return features, names    

def compile_entity_name_index(names):
    """
    Create an {entity name: index} mapping for caption encoding.
    
    Each entity name in the entity context is mapped to a unique integer index.
    
    :param names: names of entities from the entity context
    :return: the {entity name: index} mapping
    """
    # Prepare an entity name mapping for encoding captions.
    int_name_to_index = {}
    for obj in names:
        int_name_list = tuple(obj[2:])
        ind = obj[0]
        # If we've already encountered this name, skip.
        if int_name_list not in int_name_to_index:
            int_name_to_index[int_name_list] = ind
    return int_name_to_index
    
def normalize_name(name):
    """
    Normalize an input name.
    
    Normalization includes various processing with the goal of unifying
    the names that belong to the same entity but are spelled or presented
    differently.
    The normalization rules have been developed with the DBpedia entities
    as the typical target. Different specific normalization rules might be needed
    for a different data source.
    
    :param name: the original name
    :return: the normalized name
    """
    name = name.lower()
    # Remove the unnecessary parts.
    split_get_last_part = ["/", "#"]
    split_get_first_part = ["_(", ",", "_of_england"]
    for x in split_get_last_part:
        name = name.split(x)[-1].strip()
    for x in split_get_first_part:
        name = name.split(x)[0].strip()
    # Replace some characters/words with others;
    # replacing with an empty string to remove.
    to_replace = [
        ("*", ""), ("|", ""), ("''", ""), ('""', ""), ('``', ""), ('"', ""),
        (" ", "_"), ("__", "_"), ("_&_", "_and_"), 
        ("railway_station", "station"), ("tube_station", "station"), 
        ("s'", "s"), ("'s", "s"), ("saint", "st"), ("st.", "st")
    ]
    for x in to_replace:
        name = name.replace(x[0], x[1])
    #  Process the beginning and the end of the string.
    name = name.lstrip("(").rstrip(")").lstrip("_").rstrip("_")
    if name.startswith("the_"):
        name = name[len("the_"):]
    # Process year/date tokens.
    # "2010-01-01" -> "2010".
    yr = re.findall(r"([0-9]{4})\-[0-9]{2}\-[0-9]{2}", name)
    if yr:
        name = yr[0]
    # "c.1987" -> "1987".
    crc_yr = re.findall(r"c\.?\s?([0-9]{4})(\-[0-9]{2}\-[0-9]{2})?", name)
    if crc_yr:
        name = crc_yr[0][0]
    return name