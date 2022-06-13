import pickle
import pandas as pd 
import os
import pdb
import ast
import spacy
nlp = spacy.load("en_core_web_sm")

collection_dir = "./data/human_experiments/"
idiom_df = pd.read_csv("./data/human_experiments/idioms_filtered.csv")
idioms_correct = set(idiom_df.loc[(idiom_df["exclude"] != 1) & (idiom_df["pos_wrong"] != 1)]["ngram"])

all_bigrams_df = pd.read_csv("./data/human_experiments/bigrams_filtered.csv")
all_bigrams_correct = set(all_bigrams_df.loc[(all_bigrams_df["exclude"] != 1) & (all_bigrams_df["pos_wrong"] != 1)]["ngram"])

if __name__ == "__main__": 
    samples_all = {}
    assoc_idiom = []
    ngram = []
    freq = []
    unique_keys = set()
    use_all = False

    for filename in os.listdir(collection_dir):
        if not filename.endswith(".p"):
            continue
        if filename != "similar_sample_idioms.p" and filename != "similar_sample_bigrams_new.p":
            continue

        samples = pickle.load(open(f"{collection_dir}{filename}", "rb"))
        for key in samples:
            if key not in idioms_correct and key not in all_bigrams_correct: # exclude offensive/wrongly tagged ones
                continue
            
            key_words = " ".join([x.split("/")[0] for x in ast.literal_eval(key)])
            doc = nlp(key_words)
            key_lemma = " ".join([token.lemma_ for token in doc])

            if key_lemma in unique_keys:
                continue
            
            unique_keys.add(key_lemma)
            samples[key] = [x for x in samples[key] if x[1] != key]
            samples[key].sort(key=lambda x: x[0], reverse=True)

            matched_samples = samples[key] if use_all else samples[key][:3]
            if key not in samples_all:
                samples_all[key] = set(matched_samples)
            else:
                samples_all[key].union(set(matched_samples))

            ngrams = [x[1] for x in samples[key]]
            freqs = [x[2] for x in samples[key]]

            assoc_idiom.extend([key] * len(matched_samples))
            
            if use_all:
                ngram.extend(ngrams)
                freq.extend(freqs)
            else:
                ngram.extend(ngrams[:3])
                freq.extend(freqs[:3])

    assoc_df = pd.DataFrame({"assoc_idiom": assoc_idiom, "ngram": ngram, "freq": freq})
    assoc_df = assoc_df.sort_values(by="assoc_idiom")
    assoc_df.to_csv(f"{collection_dir}/random_sample_filtered_3_match.csv")
