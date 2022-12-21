import json
import pandas as pd
import pickle
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

import datasets as data
import jensen_shannon_metric as js
import utils as ut


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # sets device for model and PyTorch tensors
cudnn.benchmark = True

# Specify pointers to the data.
data_dir = "img_caption_data/input_dataset_files/"  # folder with the input data files
to_base_name = "_georic2"
data_name = "geo_aware" + to_base_name
checkpoint_name = f"checkpoint_108_{data_name}.pth.tar"  # model checkpoint
word_map_file = f"{data_dir}/WORDMAP_{data_name}.json"  # word map

# Load the trained model.
checkpoint = torch.load(checkpoint_name, map_location=torch.device(device))
decoder = checkpoint["decoder"]
decoder.to(device)
decoder.eval()
encoder = checkpoint["encoder"]
encoder.to(device)
encoder.eval()

# Load the word map.
with open(word_map_file, "r") as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

def evaluate(js_metric, max_caption_len):
    """
    Generate captions for the test set.
    
    :param js_metric: the metric to measure the accuracy of the generated geographic references
    :param max_caption_len: the maximum caption length to generate
    """
    # Define the DataLoader.
    loader = torch.utils.data.DataLoader(
        data.CaptionDataset(
            data_dir,
            data_name,
            "TEST",
            transform=transforms.Compose([normalize]),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Initialize a list to store all the generated captions.
    all_generated_captions = []
    # Iterate over images.
    for i, (image, _, _, _, entity_features, entity_names) in enumerate(tqdm(loader)):
        #
        # Move to GPU, if possible.
        image = image.to(device)  # (1, 3, 256, 256)
        entity_names = entity_names.to(device)
        #
        # Encode the image.
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)  # (1, num_pixels, encoder_dim)
        # Generate a caption.
        seq = decoder.predict(encoder_out, max_caption_len, entity_features)
        #
        # Convert the predicted token indices into text.
        if type(seq) is not list:
            seq = seq.transpose(1, 0).squeeze(0)
        entity_names = entity_names.squeeze(0)
        gen_tokens = []
        for token_idx in seq:
            if type(token_idx) is not int:
                token_idx = token_idx.item()
            if token_idx >= vocab_size:
                # Entity name was generated.
                idx_in_entity_context = token_idx - vocab_size
                if idx_in_entity_context >= entity_names.shape[0]:
                    gen_tokens.append("<unk_ent>")
                else:
                    int_name = entity_names[idx_in_entity_context][2:].tolist()
                    len_name = entity_names[idx_in_entity_context][1].item()
                    gen_tokens.append(ut.int_to_str(int_name, len_name))
            else:
                # Regular vocabulary word was generated.
                if token_idx not in {
                    word_map["<start>"],
                    word_map["<end>"],
                    word_map["<pad>"],
                }:
                    gen_tokens.append(rev_word_map[token_idx])
        generated_caption = " ".join(gen_tokens)
        # Clean up the caption.
        if not generated_caption.endswith(".") and generated_caption.count(".") > 1:
            generated_caption = ".".join(generated_caption.split(".")[:-1]) + "."
        all_generated_captions.append(generated_caption)
        # Update the Jensen-Shannon metric.
        js_metric.run(seq, entity_features.squeeze(0), entity_names.squeeze(0))
    
    print()
    print(checkpoint_name)
    # Save the generated captions.
    df_generated_captions = pd.DataFrame({"generated_caption": all_generated_captions})
    df_generated_captions.to_csv("generated_captions.csv", index=False)
    #
    # Compute the accuracy of the generated geographic references.
    js_metric.results()

if __name__ == "__main__":
    # Initialize the geo reference accuracy checker (the Jensen-Shannon metric).
    js_metric = js.JSGeoMetric(word_map = word_map)
    # Run the evaluation.
    evaluate(js_metric, max_caption_len=30)
