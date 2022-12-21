import json
import pandas as pd

# Adapted from https://github.com/tylin/coco-caption.
from evalfunc.bleu.bleu import Bleu
from evalfunc.cider.cider import Cider
from evalfunc.meteor.meteor import Meteor
from evalfunc.rouge.rouge import Rouge

# Load the generated captions for the images from the test set.
generated_captions_data = pd.read_csv("generated_captions.csv")["generated_caption"].values
# Load the ground truth captions for the images from the test set.
with open("img_caption_data/captions_split.json", "r") as j:
    data = json.load(j)
test_img_data = []
for img in data["images"]:
    if img["split"] == "test":
        test_img_data.append(img)
#
true_captions = []
generated_captions = []
urls = []
for i, img in enumerate(test_img_data):
    true_caption = " ".join(img["tokens"]).replace("_", " ")
    true_captions.append(true_caption)
    generated_captions.append(generated_captions_data[i].replace("_", " "))
    urls.append(img["url"])

# Get the metric scores.
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
all_scores["url"] = urls
all_scores["true_caption"] = true_captions
all_scores["generated_caption"] = generated_captions
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