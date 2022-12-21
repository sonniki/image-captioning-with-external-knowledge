import json
import pandas as pd
import pickle
import re

import utils as ut

class FactAccuracyMetric:
    def __init__(self, splits_captions_json_path, entity_context_path, knowledge_path):
        """
        Initialize.
        
        :param splits_captions_json_path: path to the JSON file with the caption data split into train-val-test sets
        :param entity_context_path: path to the pickle file with the entity context for each image
        :param knowledge_path: path to the pickle file with the knowledge context for each image
        """
        # Load the file with the captions data.
        with open(splits_captions_json_path, "r") as j:
            self.data = json.load(j)
        # Load the entity context and the knowledge context.
        with open(entity_context_path, "rb") as f:
            entity_contexts = pickle.load(f)
        with open(knowledge_path, "rb") as f:
            knowledge_contexts = pickle.load(f)
        # Normalize names in the entity/knowledge contexts to unify them for comparison.
        for url in entity_contexts:
            entity_context = entity_contexts[url]
            knowledge_context = knowledge_contexts[url]
            entity_context["name"] = entity_context["name"].apply(ut.normalize_name)
            knowledge_context["subject"] = knowledge_context["subject"].apply(ut.normalize_name)
            entity_contexts[url] = entity_context
            knowledge_contexts[url] = knowledge_context
        self.entity_contexts = entity_contexts
        self.knowledge_contexts = knowledge_contexts
        # Load the predicates, where synonymous ones are "merged", i.e. mapped to a single label.
        # E.g. both "opened" and "openingyear" are mapped to "opened".
        with open("data/predicates_merged_synonyms.pkl", "rb") as f:
            self.predicates_merged_synonyms = pickle.load(f)
        # Load the mapping between certain entity types and predicates that are synonymous for this
        # entity type but not necessarily for others. E.g. for "bridge", predicates "built" and "opened" are
        # largely synonymous in our corpus, since they are used interchangeably with the same years.
        with open("data/predicates_merged_for_entity_type.pkl", "rb") as f:
            self.predicates_merged_for_entity_type = pickle.load(f)
        # Load the mapping from common predicates to typical phrases, with which they are
        # realized in texts, e.g. "built" is mapped to "built in", "constructed in", etc.
        with open("data/predicate_to_phrases.pkl", "rb") as f:
            self.predicate_to_phrases = pickle.load(f)
            
    def run(self, generated_captions):
        """
        Run the factual accuracy metric.
        
        :param generated_captions: the captions to run the metric for
        """
        # Get the ground truth captions and related data for the same image.
        ground_truth_captions, urls, gt_entity_names_in_captions = self.get_ground_truth_data(generated_captions)
        #
        # Check the generated facts.
        generated_facts = {
            "temporal": [], "correct_temporal": [],
            "other": [], "correct_other": [],
        }
        for i in range(len(generated_captions)):
            generated_caption = generated_captions[i]
            ground_truth_caption = ground_truth_captions[i]
            gt_entity_names_in_caption = gt_entity_names_in_captions[i]
            url = urls[i]
            # Check temporal facts.
            has_fact, has_correct_fact = self.check_temporal_facts(
                generated_caption, ground_truth_caption, gt_entity_names_in_caption, url
            )
            generated_facts["temporal"].append(has_fact)
            generated_facts["correct_temporal"].append(has_correct_fact)
            # Check other types of facts.
            has_fact, has_correct_fact = self.check_other_facts(
                generated_caption, gt_entity_names_in_caption, url
            )
            generated_facts["other"].append(has_fact)
            generated_facts["correct_other"].append(has_correct_fact)
        # Print the results.
        acc_temporal = 0.0 if sum(generated_facts['temporal']) == 0 else sum(generated_facts['correct_temporal']) / sum(generated_facts['temporal'])
        acc_other = 0.0 if sum(generated_facts['other']) == 0 else sum(generated_facts['correct_other']) / sum(generated_facts['other'])
        all_generated_facts_count = sum(generated_facts['other']) + sum(generated_facts['temporal'])
        if all_generated_facts_count == 0:
            acc_all = 0.0
        else:
            acc_all = (sum(generated_facts['correct_other']) + sum(generated_facts['correct_temporal'])) / all_generated_facts_count
        print(f"Accuracy (temporal): {acc_temporal}")
        print(f"Accuracy (other): {acc_other}")
        print(f"ACCURACY (all): {acc_all}")
    
    def check_temporal_facts(self, generated_caption, ground_truth_caption, gt_entity_names_in_caption, url):
        """
        Check the accuracy of temporal facts in one caption.
        
        :param generated_caption: the generated caption
        :param ground_truth_caption: the ground truth caption for the same image
        :param gt_entity_names_in_caption: the entities found in the ground truth caption
        :param url: the URL of the image
        :return: 
            - whether the generated caption contains temporal facts
            - whether the generated caption contains correct temporal facts
        """
        entity_context = self.entity_contexts[url]
        knowledge_context = self.knowledge_contexts[url]
        generated_years = self.extract_year(generated_caption)
        if not generated_years:
            has_fact = False
            has_correct_fact = False
        else:
            has_fact = True
            correct_entity_name = self.check_entity_name_for_year(
                generated_caption, generated_years, gt_entity_names_in_caption, knowledge_context, entity_context
            )
            correct_predicate = self.check_predicate_for_year(
                generated_caption, generated_years, ground_truth_caption, knowledge_context
            )
            has_correct_fact = correct_entity_name and correct_predicate
        return has_fact, has_correct_fact
    
    @staticmethod
    def check_entity_name_for_year(generated_caption, generated_years, gt_entity_names_in_caption, knowledge_context, entity_context):
        """
        Check whether the years have been generated with the relevant entities.
        
        I.e. check that the generated years appear as objects in the knowledge context facts,
        the subjects of which are the entities that are relevant to the image and have
        also been generated in the caption.
        
        :param generated_caption: the generated caption
        :param generated_years: the years found in the generated caption
        :param gt_entity_names_in_caption: the entities found in the ground truth caption
        :param knowledge_context: the knowledge context for the image
        :param entity_context: the entity context for the image
        :return: whether the years have been generated with the relevant entities
        """
        # Get the subjects of the facts, the objects of which are the generated years.
        generated_year_subjects = list(knowledge_context[knowledge_context["object"].isin(generated_years)]["subject"].unique())
        # Select the ones that are relevant to the image as confirmed by the ground truth caption.
        entities_for_generated_years = []
        for subj in generated_year_subjects:
            if subj in gt_entity_names_in_caption:
                entities_for_generated_years.append(subj)
            else:
                longest_match_name = ""
                longest_match_len = 0
                for entity_name in gt_entity_names_in_caption:
                    if (
                        (subj in entity_name and "_" in subj) or 
                        (entity_name in subj and "_" in entity_name)
                    ):
                        if len(entity_name) > longest_match_len:
                            longest_match_name = entity_name
                            longest_match_len = len(entity_name)
                if longest_match_name != "":
                    entities_for_generated_years.append(longest_match_name)
        # Get the entities from the entity context that correspond to the relevant fact subjects.
        entity_context_for_subjects = entity_context[entity_context["name"].isin(generated_year_subjects)]
        if not entity_context_for_subjects.empty:
            entities_for_generated_years = [x for x in entity_context_for_subjects["name"].values]
        else:
            entities_for_generated_years = []
            for subj in generated_year_subjects:
                longest_match_name = "unk"
                longest_match_len = 0
                for entity_name in entity_context["name"].values:
                    if (
                        (subj in entity_name and "_" in subj) or 
                        (entity_name in subj and "_" in entity_name)
                    ):
                        if len(entity_name) > longest_match_len:
                            longest_match_name = entity_name
                            longest_match_len = len(entity_name)
                if longest_match_name != "unk":
                    entities_for_generated_years.append(longest_match_name)
        # Check whether the entities have been generated in the caption and are relevant to the image.
        for name in entities_for_generated_years:
            if (
                (name in generated_caption or name.replace(" ", "_") in generated_caption) and 
                (
                    name.replace(" ", "_") in gt_entity_names_in_caption or 
                    any(name.replace(" ", "_") in n for n in gt_entity_names_in_caption) or 
                    any(n in name.replace(" ", "_") for n in gt_entity_names_in_caption)
                )
            ):
                return True
        return False
        
    def check_predicate_for_year(self, generated_caption, generated_years, ground_truth_caption, knowledge_context):
        """
        Check whether the years have been generated with the correct predicate phrases.
        
        I.e. check that the generated years appear as objects in the knowledge context facts,
        the predicates of which have been realized in the generated caption as well.
        
        :param generated_caption: the generated caption
        :param generated_years: the years found in the generated caption
        :param ground_truth_caption: the ground truth caption
        :param knowledge_context: the knowledge context for the image
        :return: whether the years have been generated with the correct predicate phrases
        """
        # Get the predicates of the facts where the generated years are objects.
        generated_year_facts = knowledge_context[knowledge_context["object"].isin(generated_years)]
        subject_predicate_tags = generated_year_facts.groupby(["subject", "predicate"]).cumcount()
        generated_year_facts = generated_year_facts.assign(subject_predicate_num_idx = subject_predicate_tags)
        generated_year_facts["predicate"] = generated_year_facts.apply(
            lambda row: row["predicate"] + "_" + str(row["subject_predicate_num_idx"]) if row["predicate"] == "years" else row["predicate"], axis=1
        )
        generated_predicates = generated_year_facts["predicate"].unique()
        for generated_predicate in generated_predicates:
            # If the predicate is in a group of synonymous ones that have been manually mapped to a single label, 
            # use the label instead of the original predicate.
            if generated_predicate in self.predicates_merged_synonyms:
                predicate = self.predicates_merged_synonyms[generated_predicate]
            else:
                predicate = generated_predicate
            # Get the phrases which can realize the predicate in the caption.
            if predicate in self.predicate_to_phrases:
                predicate_phrases = self.predicate_to_phrases[predicate]["phrases"]
            else:
                predicate_phrases = [predicate.replace("_", " ")]
            if any(phrase in generated_caption for phrase in predicate_phrases):
                return True
            # For some entity types, certain predicates are synonymous even if they are not necessarily synonymous otherwise.
            # In these cases, predicate phrases are combined.
            for entity_type in self.predicates_merged_for_entity_type:
                if entity_type in generated_caption and predicate in self.predicates_merged_for_entity_type[entity_type]:
                    for predicate in self.predicates_merged_for_entity_type[entity_type]:
                        predicate_phrases.extend(self.predicate_to_phrases[predicate]["phrases"])
                    if any(phrase in generated_caption for phrase in predicate_phrases):
                        return True
        # Check if the generated and the ground truth caption contain predicate phrases of the same category for the same years.
        ground_truth_years = self.extract_year(ground_truth_caption)
        same_years = list(set(ground_truth_years) & set(generated_years))
        if not len(same_years):
            return False
        for same_year in same_years:
            left_context_ground_truth = ground_truth_caption[: ground_truth_caption.find(same_year)]
            left_context_generated = generated_caption[: generated_caption.find(same_year)]
            for predicate, predicate_data in self.predicate_to_phrases.items():
                if predicate_data["type"] != "temporal":
                    continue
                if (
                    any(phrase in left_context_generated for phrase in predicate_data["phrases"]) and
                    any(phrase in left_context_ground_truth for phrase in predicate_data["phrases"])
                ):
                    return True
        return False
            
    def check_other_facts(self, generated_caption, gt_entity_names_in_caption, url):
        """
        Check the accuracy of non-temporal facts in one caption.
        
        :param generated_caption: the generated caption
        :param gt_entity_names_in_caption: the entities found in the ground truth caption
        :param url: the URL of the image
        :return: 
            - the number of generated non-temporal facts
            - the number of generated correct non-temporal facts
        """
        # Get the entity names that are relevant to the image and have been generated in the caption.
        entity_context = self.entity_contexts[url]
        generated_entity_names = []
        for entity_name in entity_context["name"].values:
            if entity_name in generated_caption or entity_name.replace("_", " ") in generated_caption:
                if (
                    entity_name.replace(" ", "_") in gt_entity_names_in_caption or 
                    any(n in entity_name.replace(" ", "_") for n in gt_entity_names_in_caption) or 
                    any(entity_name.replace(" ", "_") in n for n in gt_entity_names_in_caption)
                ):
                    generated_entity_names.append(entity_name)
        # Get the fact subjects that correspond to these entity names.
        knowledge_context = self.knowledge_contexts[url]
        generated_fact_subjects = []
        for name in generated_entity_names:
            for subj in knowledge_context["subject"].unique():
                if subj in name or name in subj:
                    generated_fact_subjects.append(subj)
        #
        # Locate the non-temporal facts and check their accuracy. 
        has_fact = 0
        has_correct_fact = 0
        all_possible_objects = knowledge_context["object"].unique()
        if not any(x in generated_caption for x in all_possible_objects):
            # No facts have been generated at all.
            return has_fact, has_correct_fact
        # Iterate over predicates.
        for predicate, predicate_data in self.predicate_to_phrases.items():
            if predicate_data["type"] == "temporal":
                    continue
            # Extend some predicates with their synonyms (e.g. "architect" + "designer").
            # Here, synonyms are predicates that can be used interchangeably with the same objects in our data.
            predicate_plus_synonyms = [predicate] + predicate_data["synonymous_predicates"]
            facts_predicate = knowledge_context[knowledge_context["predicate"].isin(predicate_plus_synonyms)]
            # Select the facts with this predicate and the entities generated in the caption.
            facts_predicate_enitity = facts_predicate[facts_predicate["subject"].isin(generated_fact_subjects)]
            # Get the fact objects that can potentially appear in the caption.
            expected_objects = facts_predicate_enitity["object"].unique()
            # 
            for phrase in predicate_data["phrases"]:
                if type(phrase) == str:
                    # Check that 1) the caption contains the predicate phrase, 2) the caption contains the additional tokens indicating
                    # the correct entity type, if needed, 3) the caption doesn't contain any of the phrases clashing with the predicate.
                    if (
                        phrase in generated_caption and 
                        (len(predicate_data["entity_types"]) == 0 or any(ent in generated_caption for ent in predicate_data["entity_types"])) and
                        not any(p in generated_caption for p in predicate_data["blocklist"])
                    ):
                        # Get the part of the caption to look for the fact objects in.
                        if predicate_data["object_position"] == "right":
                            caption_context = phrase.join(generated_caption.split(phrase)[1:])
                        else:
                            caption_context = generated_caption
                        # Record whether the caption contains a fact with this predicate and whether this fact is correct.
                        if any(obj in caption_context for obj in all_possible_objects):
                            has_fact += 1
                            if any(obj in caption_context for obj in expected_objects):
                                has_correct_fact += 1
                elif type(phrase) == tuple:
                    # Check that 1) the caption contains a multi-token predicate phrase where tokens can be separated by other tokens,
                    # 2) the caption contains the additional tokens indicating the correct entity type, if needed, 3) the caption doesn't contain 
                    # any of the phrases clashing with the predicate.
                    if (
                        phrase[0] in generated_caption and phrase[1] in generated_caption and 
                        (generated_caption.find(phrase[0]) - generated_caption.find(phrase[1])) < 0 and
                        (generated_caption.find(phrase[1]) - generated_caption.find(phrase[0])) < 20 and
                        (len(predicate_data["entity_types"]) == 0 or (any(ent in generated_caption for ent in predicate_data["entity_types"]) and
                        (generated_caption.find(phrase[0]) - max(generated_caption.find(ent) for ent in predicate_data["entity_types"]) > 0))) and
                        not any(p in generated_caption for p in predicate_data["blocklist"])
                    ):
                        # Get the part of the caption to look for the fact objects in.
                        if predicate_data["object_position"] == "right":
                            caption_context = phrase[1].join(generated_caption.split(phrase[1])[1:])
                        else:
                            caption_context = generated_caption
                        # Record whether the caption contains a fact with this predicate and whether this fact is correct.
                        if any(obj in caption_context for obj in all_possible_objects):
                            has_fact += 1
                            if any(obj in caption_context for obj in expected_objects):
                                has_correct_fact += 1
        return has_fact, has_correct_fact
        
    def get_ground_truth_data(self, generated_captions):
        """
        Get the ground truth caption data for the same images that the generated captions were produced for.
        
        :param generated_captions: the automatically generated captions
        :return: 
          - the ground truth captions for the same images
          - the URLs of these images
          - the entity names present in the ground truth captions
        """
        urls = []
        ground_truth_captions = []
        ground_truth_entity_names = []
        for img in self.data["images"]:
            if img["split"] != "test":
                # Only consider images from the test set.
                continue
            ground_truth_caption = " ".join(img["tokens"])
            cur_ground_truth_entity_names = []
            for i, token in enumerate(img["tokens"]):
                if img["mask"][i] == 1:
                    # Store entity names found in the caption.
                    cur_ground_truth_entity_names.append(token)
            ground_truth_captions.append(ground_truth_caption)
            ground_truth_entity_names.append(cur_ground_truth_entity_names)
            urls.append(img["url"])
        assert len(ground_truth_captions) == len(generated_captions) == len(urls) == len(ground_truth_entity_names)
        return ground_truth_captions, urls, ground_truth_entity_names
    
    @staticmethod
    def extract_year(text):
        """
        Extract years from the input text.
        
        Only years between 1000 and 2000 are considered.
        Since "<unk_fact>" can also be generated in the same position as a year, e.g. "build in <unk_fact>",
        it is included as well (even though during fact accuracy evaluation it is going to be marked as incorrect).
        
        :param text: the text to extract years from
        :return: the extracted years
        """
        year_regex = r"\b(1\d{3})\b"
        year_regex = r"(?<!footpath\s)(?<!postbox )" + year_regex
        year_regex = r"(?<!no.)(?<!no.\s)(?<!no\s)" + year_regex
        year_regex = r"(?<!no\..{4}\s)(?<!no\s.{4}\s)(?<!no\.\s.{4}\s)" + year_regex
        year_regex = r"(?i)" + year_regex
        years = re.findall(year_regex, text)
        years_unk = re.findall("<unk_fact>", text)
        return years + years_unk
    
    @staticmethod
    def is_year(text):
        """
        Determine if the input text could be a year.
        
        E.g. "1984" can be a year, "random text" or "123" cannot. 
        Only years between 1000 and 2000 are considered.
        
        :param text: the text to check
        :return: whether the text can be a year
        """
        try:
            int(text)
        except ValueError:
            return False
        year_reg = r"\b(1\d{3})\b"
        if len(re.findall(year_reg, text)):
            return True
        return False
    