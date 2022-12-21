import json
import pandas as pd
import pickle
import random
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm

import datasets as data
import fact_accuracy_metric as fm
import utils as ut


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # sets device for model and PyTorch tensors
cudnn.benchmark = True

# Specify pointers to the data.
data_dir = "img_caption_data/input_dataset_files/"  # folder with the input data files
to_base_name = ""
data_name = "knowledge_from_metadata" + to_base_name
checkpoint_name = f"checkpoint_{data_name}.pth.tar"  # model checkpoint
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
# Load the dict with {predicate: possible objects for the predicate} data (for the random fact object baseline). 
with open(f"data/predicate_to_objects.pkl", "rb") as f:
    predicate_to_objects = pickle.load(f)

def evaluate(fact_accuracy_metric, max_caption_len):
    """
    Generate captions for the test set.
    
    :param fact_accuracy_metric: the metric to measure the accuracy of the generated facts
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
    all_generated_captions_random_baseline = []
    # Iterate over images.
    for i, (image, _, _, _, entity_features, entity_names, facts, fact_names) in enumerate(tqdm(loader)):
        #
        # Move to GPU, if possible.
        image = image.to(device)  # (1, 3, 256, 256)
        entity_names = entity_names.to(device)
        facts = facts.to(device)
        fact_names = fact_names.to(device)
        # Prepare the data for the random fact object baseline.
        years_in_knowledge_context = []
        other_objects_in_knowledge_context = []
        for idx in range(fact_names.shape[1]-1):
            int_name = fact_names[0][idx][2:].tolist()
            len_name = fact_names[0][idx][1].item()
            fact_obj = ut.int_to_str(int_name, len_name)
            if fact_obj not in other_objects_in_knowledge_context and fact_obj not in years_in_knowledge_context:
                if fact_metric.is_year(fact_obj):
                    years_in_knowledge_context.append(fact_obj)
                else:
                    other_objects_in_knowledge_context.append(fact_obj)
        #
        # Encode the image.
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)  # (1, num_pixels, encoder_dim)
        # Generate a caption.
        seq = decoder.predict(encoder_out, max_caption_len, entity_features, facts)
        #
        # Convert the predicted token indices into text.
        if type(seq) is not list:
            seq = seq.transpose(1, 0).squeeze(0)
        entity_names = entity_names.squeeze(0)
        fact_names = fact_names.squeeze(0)
        gen_tokens = []
        gen_tokens_random_baseline = []
        for token_idx in seq:
            if type(token_idx) is not int:
                token_idx = token_idx.item()
            if token_idx >= vocab_size and token_idx < vocab_size + entity_names.shape[0]:
                # Entity name was generated.
                idx_in_entity_context = token_idx - vocab_size
                if idx_in_entity_context >= entity_names.shape[0]:
                    gen_tokens.append("<unk_ent>")
                    gen_tokens_random_baseline.append("<unk_ent>")
                else:
                    int_name = entity_names[idx_in_entity_context][2:].tolist()
                    len_name = entity_names[idx_in_entity_context][1].item()
                    gen_tokens.append(ut.int_to_str(int_name, len_name))
                    gen_tokens_random_baseline.append(ut.int_to_str(int_name, len_name))
            elif token_idx >= vocab_size + entity_names.shape[0]:
                # Fact token was generated.
                idx_in_knowledge_context = token_idx - vocab_size - entity_names.shape[0]
                if idx_in_knowledge_context >= fact_names.shape[0]:
                    gen_tokens.append("<unk_fact>")
                    gen_tokens_random_baseline.append("<unk_fact>")
                else:
                    int_name = fact_names[idx_in_knowledge_context][2:].tolist()
                    len_name = fact_names[idx_in_knowledge_context][1].item()
                    gen_fact_token = ut.int_to_str(int_name, len_name)
                    gen_tokens.append(gen_fact_token)
                    # For the random fact object baseline, randomly select a fact token of the same type from the knowledge context.
                    if not fact_metric.is_year(gen_fact_token):
                        if not len(other_objects_in_knowledge_context):
                            random_fact = "<unk_fact>"
                        else:
                            same_type_fact_tokens = [gen_fact_token]
                            for pred in predicate_to_objects:
                                objects = predicate_to_objects[pred]
                                if gen_fact_token in objects:
                                    # Whether or not the type of the fact token is the same is determined by whether or not it can appear 
                                    # as an object with the same predicate in our corpus.
                                    same_type_fact_tokens.extend([x for x in objects if x != gen_fact_token and x in other_objects_in_knowledge_context])
                            if len(same_type_fact_tokens):
                                random_fact = random.choice(same_type_fact_tokens)
                            else:
                                random_fact = "<unk_fact>"
                    else:
                        if len(years_in_knowledge_context):
                            random_fact = random.choice(years_in_knowledge_context)
                        else:
                            random_fact = "<unk_fact>"
                    gen_tokens_random_baseline.append(random_fact)
            else:
                # Regular vocabulary word was generated.
                if token_idx not in {
                    word_map["<start>"],
                    word_map["<end>"],
                    word_map["<pad>"],
                }:
                    gen_tokens.append(rev_word_map[token_idx])
                    gen_tokens_random_baseline.append(rev_word_map[token_idx])
        generated_caption = " ".join(gen_tokens)
        generated_caption_random_baseline = " ".join(gen_tokens_random_baseline)
        # Clean up the caption.
        if not generated_caption.endswith(".") and generated_caption.count(".") > 1:
            generated_caption = ".".join(generated_caption.split(".")[:-1]) + "."
            generated_caption_random_baseline = ".".join(generated_caption_random_baseline.split(".")[:-1]) + "."
        #
        all_generated_captions.append(generated_caption)
        all_generated_captions_random_baseline.append(generated_caption_random_baseline)
    
    print()
    print(checkpoint_name)
    # Save the generated captions.
    df_generated_captions = pd.DataFrame({"generated_caption": all_generated_captions})
    df_generated_captions.to_csv("generated_captions.csv", index=False)
    # Save the captions for the random fact baseline.
    df_generated_captions_random = pd.DataFrame({"generated_caption": all_generated_captions_random_baseline})
    df_generated_captions_random.to_csv("generated_captions_random_facts.csv", index=False)
    #
    # Compute the fact accuracy of the generated captions.
    print("\nKNOWLEDGE-AWARE:")
    fact_metric.run(all_generated_captions)
    print("\n-----------------------------------------------------------------------------")
    print("\nRANDOM FACT OBJECT BASELINE:")
    fact_metric.run(all_generated_captions_random_baseline)

if __name__ == "__main__":
    # Initialize the fact accuracy checker.
    fact_metric = fm.FactAccuracyMetric(
        splits_captions_json_path="img_caption_data/captions_split.json",
        entity_context_path="img_caption_data/entity_context.pkl",
        knowledge_path="img_caption_data/knowledge_context.pkl",
    )
    # Run the evaluation.
    evaluate(fact_metric, max_caption_len=40)
