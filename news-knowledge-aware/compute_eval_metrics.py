import json
import pandas as pd
import spacy
import truecase
from tqdm import tqdm

# Adapted from https://github.com/tylin/coco-caption.
from evalfunc.bleu.bleu import Bleu
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor
from evalfunc.rouge.rouge import Rouge

nlp = spacy.load("en_core_web_sm")
def get_entities(text):
    doc = nlp(text)
    res_entities = [x.text for x in doc.ents]
    res_entities = sorted(set(res_entities))
    return res_entities

# Load the generated captions for the images from the test set.
generated_captions_data = pd.read_csv("generated_captions.csv")["generated_caption"].values
generated_masks_data = pd.read_csv("generated_captions.csv")["generated_mask"].values
# Load the ground truth captions for the images from the test set.
with open("img_caption_data/captions_split.json", "r") as j:
    data = json.load(j)
test_img_data = []
max_len = 50 # the same value as used during the creation of input files.
for img in data["images"]:
    if img["split"] == "test" and len(img["tokens"]) <= max_len:
        test_img_data.append(img)
#
true_captions = []
generated_captions = []
generated_captions_orig = [] # keeping for analysis (to preserve the mapping with the generated mask values).
generated_masks = []
ids = []
for i, img in enumerate(test_img_data):
    true_captions.append(" ".join(img["tokens"]).replace("_", " "))
    generated_captions.append(generated_captions_data[i].replace("_", " "))
    generated_captions_orig.append(generated_captions_data[i])
    generated_masks.append(generated_masks_data[i])
    ids.append(img["id"])
    
##################################################################
### PRECISION-RECALL OF NAMED ENTITIES ###########################
##################################################################

num_unique_gen_ents = []
for mode in ["exact", "partial"]:
    tp_entities = 0
    fp_entities = 0
    fn_entities = 0
    for i, gen_cap in tqdm(enumerate(generated_captions)):
        # Restore the original case of the words in the captions.
        gen_cap_recased = truecase.get_true_case(gen_cap)
        true_cap_recased = truecase.get_true_case(true_captions[i])
        # Extract the named entities.
        entities_gen = [x.lower() for x in get_entities(gen_cap_recased)]
        num_unique_gen_ents.append(len(entities_gen))
        entities_true = [x.lower() for x in get_entities(true_cap_recased)]
        for ent in entities_true:
            if mode == "exact" and (ent in entities_gen or ent in gen_cap):
                tp_entities += 1
            elif mode == "partial" and (
                any(ent in e for e in entities_gen) or 
                any(e in ent for e in entities_gen) or 
                any(token in gen_cap for token in ent.split())
            ):
                tp_entities += 1
            else:
                fn_entities += 1
        for ent in entities_gen:
            if mode == "exact" and ent not in entities_true and ent not in true_captions[i]:
                fp_entities += 1
            elif mode == "partial":
                if (
                    not any(ent in e for e in entities_true) and 
                    not any(e in ent for e in entities_true) and 
                    not any(token in true_captions[i] for token in ent.split())
                ):
                    fp_entities += 1
    precision = tp_entities / (tp_entities + fp_entities) if (tp_entities + fp_entities) != 0 else 0
    recall = tp_entities / (tp_entities + fn_entities) if (tp_entities + fn_entities) != 0 else 0
    print()
    print(mode.capitalize() + ":")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
print()
print(f"Overall number of generated unique entities per caption: {sum(num_unique_gen_ents)/len(num_unique_gen_ents) if len(num_unique_gen_ents) else 0}")

##################################################################
### STANDARD METRIC SCORES #######################################
##################################################################

scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Cider(), "CIDEr"),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
]
score = []
method = []
all_scores = {
    "Bleu_1": [],
    "Bleu_2": [],
    "Bleu_3": [],
    "Bleu_4": [],
    "CIDEr": [],
    "METEOR": [],
    "ROUGE_L": [],
}
all_scores["id"] = ids
all_scores["true_caption"] = true_captions
all_scores["generated_caption"] = generated_captions
all_scores["generated_caption_orig"] = generated_captions_orig
all_scores["generated_mask"] = generated_masks
# Compute the metric scores.
for scorer, method_i in scorers:
    score_i, scores_i = scorer.compute_score(
        [[x] for x in true_captions], [[x] for x in generated_captions]
    )
    score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
    method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
    if isinstance(method_i, str):
        all_scores[method_i].extend(scores_i)
    else:
        # For different BLEU scores.
        for i in range(len(method_i)):
            all_scores[method_i[i]].extend(scores_i[i])
score_dict = dict(zip(method, score))
# Package the scores for captions (for analysis).
metric_scores = pd.DataFrame(all_scores)
metric_scores.to_csv("metric_scores_for_generated_captions.csv", index=False)
# Print the metric scores for the corpus.
print("\nMetric scores:\n")
for method, score_ in score_dict.items():
    print("%s score is %.4f." % (method, score_))