import math
import numpy as np
import pickle
import random

import utils as ut

class JSGeoMetric:
    def __init__(self, word_map, print_metrics=True):
        self.print_metrics = print_metrics
        self.word_map = word_map
        self.rev_word_map = {v: k for k, v in word_map.items()}
        # Load the probability distributions of geo features in the ground truth captions in train set.
        with open("data/geo_probability_distr_train.pkl", "rb") as f:
            self.geo_probability_distr_train = pickle.load(f)        
        # Load the bins for the distance values.
        with open("data/bins_distance.pkl", "rb") as f:
            self.bins_distance = pickle.load(f)
        # Load the bins for the azimuth values.
        with open("data/bins_azimuth.pkl", "rb") as f:
            self.bins_azimuth = pickle.load(f)
        # Load the entity types from OpenStreetMap.
        with open("data/OSM_types_index.pkl", "rb") as f:
            self.OSM_types_index = pickle.load(f)
        # Get the indices of spatial prepositions in the vocabulary.
        if "north_of" not in self.word_map:
            self.azimuth_words = ["north", "south", "east", "west"]
        else:
            self.azimuth_words = ["north_of", "south_of", "east_of", "west_of"]
        self.geoterm_wordmap_indices = [
            self.word_map[x]
            for x in ["near", "in", "across", "along"] + self.azimuth_words
            if x in self.word_map
        ]
        # Initialize the dicts for storing the data.
        self.geo_probability_distr_generated = {
            "near": {"n_occurrences": 0, "distance": [], "distance_probs": []},
            "along": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "across": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "in": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "north": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "south": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "east": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "west": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
        }
        self.geo_probability_distr_random = {
            "near": {"n_occurrences": 0, "distance": [], "distance_probs": []},
            "along": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "across": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "in": {"n_occurrences": 0, "distance": [], "distance_probs": [], "type": [], "type_probs": []},
            "north": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "south": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "east": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
            "west": {"n_occurrences": 0, "azimuth": [], "azimuth_probs": []},
        }
        
    ##############################################################################
    # STORING THE DATA
    ##############################################################################

    def run(self, seq, entity_context, entity_context_names):
        for i, token_idx in enumerate(seq):
            if type(token_idx) is not int:
                token_idx = token_idx.item()
            if i > 0 and token_idx >= len(self.word_map):
                # An entity name was generated; check the previous word(s).
                prev_token_idxs = [seq[i - 1]]
                if i > 1:
                    prev_token_idxs.append(seq[i - 2])
                if i > 2:
                    prev_token_idxs.append(seq[i - 3])
                for pti, it in enumerate(prev_token_idxs):
                    if type(it) is not int:
                        prev_token_idxs[pti] = it.item()
                if prev_token_idxs[0] >= len(self.word_map):
                    # The previous word is also an entity name.
                    continue
                if (
                    prev_token_idxs[0] in self.geoterm_wordmap_indices or 
                    (
                        i > 1
                        and prev_token_idxs[1] in self.geoterm_wordmap_indices
                        and self.rev_word_map[prev_token_idxs[0]] in {"of", "the", "a"}
                    ) or
                    (
                        i > 2
                        and prev_token_idxs[2] in self.geoterm_wordmap_indices
                        and prev_token_idxs[1] < len(self.word_map) and self.rev_word_map[prev_token_idxs[1]] == "of"
                        and self.rev_word_map[prev_token_idxs[0]] in {"the", "a"}
                    )
                ):
                    # The generated token is an entity name following a spatial preposition.
                    if self.rev_word_map[prev_token_idxs[0]] in {"of", "the", "a"}:
                        if self.rev_word_map[prev_token_idxs[1]] == "of":
                            geo_term = self.rev_word_map[prev_token_idxs[2]]
                        else:
                            geo_term = self.rev_word_map[prev_token_idxs[1]]
                    else:
                        geo_term = self.rev_word_map[prev_token_idxs[0]]
                    if "_" in geo_term:
                        geo_term = geo_term.split("_")[0]
                    #
                    ind_in_entity_context = token_idx - len(self.word_map)
                    if ind_in_entity_context >= entity_context.shape[0]:
                        # The entity name is an "<unk_ent>", skip.
                        continue
                    # Look up the name of the generated entity name.
                    int_name = entity_context_names[ind_in_entity_context][2:].tolist()
                    len_name = entity_context_names[ind_in_entity_context][1].item()
                    geo_name = ut.int_to_str(int_name, len_name)
                    if "unk_ent" in geo_name:
                        # The entity name is an "<unk_ent>", skip.
                        continue
                    # Record the data for the current geographic reference.
                    self.geo_probability_distr_generated[geo_term]["n_occurrences"] += 1
                    self.geo_probability_distr_generated = self.store_data_for_idx(
                        self.geo_probability_distr_generated,
                        ind_in_entity_context,
                        geo_term,
                        entity_context,
                    )
                    # Record a random entity's info for the random geo-entity baseline.
                    self.geo_probability_distr_random[geo_term]["n_occurrences"] += 1
                    def get_name(x, entity_context_names):
                        int_name = entity_context_names[x][2:].tolist()
                        len_name = entity_context_names[x][1].item()
                        return ut.int_to_str(int_name, len_name)
                    non_unk_object_indices = [x for x in range(entity_context.shape[0]) if "unk_ent" not in get_name(x, entity_context_names)]
                    random_idx = random.choice(non_unk_object_indices)
                    int_name = entity_context_names[random_idx][2:].tolist()
                    len_name = entity_context_names[random_idx][1].item()
                    random_name = ut.int_to_str(int_name, len_name)
                    self.geo_probability_distr_random = (
                        self.store_data_for_idx(
                            self.geo_probability_distr_random,
                            random_idx,
                            geo_term,
                            entity_context,
                        )
                    )
                    
    def store_data_for_idx(self, analysis_dict, idx, geo_term, entity_context):
        cur_ref_dist = entity_context[idx, 1].cpu().item()
        cur_ref_azim = entity_context[idx, 2].cpu().item()
        cur_ref_type = entity_context[idx, 4].cpu().item()
        # Distance.
        if geo_term in {"near", "along", "across", "in"}:
            for bin_idx, bin_ in enumerate(self.bins_distance):
                if cur_ref_dist >= bin_[0] and cur_ref_dist < bin_[1]:
                    analysis_dict[geo_term]["distance"].append(bin_idx)
                    break
        # Azimuth.
        if geo_term in self.azimuth_words or geo_term + "_of" in self.azimuth_words:            
            for bin_idx, bin_ in enumerate(self.bins_azimuth):
                if cur_ref_azim >= bin_[0] and cur_ref_azim < bin_[1]:
                    analysis_dict[geo_term]["azimuth"].append(bin_idx)
                    break
        # Type.
        if geo_term in {"along", "across", "in"}:
            analysis_dict[geo_term]["type"].append(cur_ref_type)
        return analysis_dict
        
    ##############################################################################
    # JSD METRIC CALCULATION
    ##############################################################################

    def results(self):
        if self.print_metrics:
            print("\GEO-AWARE:\n")
            self.compute_metrics(self.geo_probability_distr_generated, self.geo_probability_distr_train)
            print("############################################")
            print("\nRANDOM GEO-ENTITY:\n")
            self.compute_metrics(self.geo_probability_distr_random, self.geo_probability_distr_train)
            print()
        with open("data/geo_probability_distr_generated.pkl", "wb") as handle:
            pickle.dump(self.geo_probability_distr_generated, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/geo_probability_distr_random.pkl", "wb") as handle:
            pickle.dump(self.geo_probability_distr_random, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_metrics(self, analysis_dict, reference_values):
        """
        Compute the JS distance metric score for each spatial preposition.
        """
        # Turn the value/bin distributions into probability distributions.
        for term in analysis_dict:
            for feature in analysis_dict[term]:
                if feature == "n_occurrences":
                    continue
                elif feature == "distance":
                    for bin_idx, bin_ in enumerate(self.bins_distance):
                        bin_count = analysis_dict[term][feature].count(bin_idx)
                        prob = float(bin_count) / analysis_dict[term]["n_occurrences"]
                        analysis_dict[term][f"{feature}_probs"].append(prob)
                elif feature == "azimuth":
                    for bin_idx, bin_ in enumerate(self.bins_azimuth):
                        bin_count = analysis_dict[term][feature].count(bin_idx)
                        prob = float(bin_count) / analysis_dict[term]["n_occurrences"]
                        analysis_dict[term][f"{feature}_probs"].append(prob)
                elif feature == "type":
                    for type_i in range(len(self.OSM_types_index)):
                        type_count = analysis_dict[term][feature].count(type_i)
                        prob = float(type_count) / analysis_dict[term]["n_occurrences"]
                        analysis_dict[term][f"{feature}_probs"].append(prob)
        # Compute the the Jensen-Shannon distance for every term.
        for term in analysis_dict:
            print(term.upper())
            # Output the number of occurrences for each term.
            print(
                f"Number of occurrences: {analysis_dict[term]['n_occurrences']}"
            )
            if analysis_dict[term]["n_occurrences"] == 0:
                print()
                continue
            for feature in analysis_dict[term]:
                if "_probs" not in feature:
                    continue
                feature_name = feature.split("_")[0]
                # Compute the Jensen-Shannon distance.
                q = np.asarray(reference_values[term][feature])
                p = np.asarray(analysis_dict[term][feature])
                jsd = self.js_distance(q, p)
                print(f"{feature_name}: {jsd}")
            print()

    def js_distance(self, p, q):
        """
        Calculate the Jensen-Shannon distance.
        """
        m = 0.5 * (p + q)
        js_divergence = 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
        js_distance = math.sqrt(js_divergence)
        return js_distance
        
    @staticmethod
    def kl_divergence(p, q):
        """
        Calculate the Kullback-Leibler divergence.
        """
        res_lst = [] 
        for i in range(len(p)):
            if q[i] == 0 or p[i] == 0:
                res_lst.append(0)
            else:
                res_lst.append(p[i] * math.log2(p[i]/q[i]))
        return sum(res_lst)
    