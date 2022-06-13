import pandas as pd
import pdb
import ast
import random

from src.data.sample_syntactic_ngrams import get_binparse_from_samples, parse_text

path = "./data/human_experiments/random_sample_filtered_3_match.csv"

if __name__ == "__main__":
    df = pd.read_csv(path)
    unique_ngrams = df["assoc_idiom"].unique()
    phrases = []
    left_texts = []
    right_texts = []
    assoc_idioms = []
    freqs = []
    for ngram in unique_ngrams:
        rows = df.loc[df["assoc_idiom"] == ngram].drop_duplicates(subset=["ngram"])
        if len(rows) == 0:
            continue
        elif len(rows) < 5:
            row_sample = rows
        else:
            row_sample = rows.sample(n=5)
        ngram_text = " ".join([x.split("/")[0] for x in ast.literal_eval(ngram)])
        left_text_idiom, right_text_idiom = parse_text(ngram_text)

        phrases.append(ngram_text)
        left_texts.append(left_text_idiom)
        right_texts.append(right_text_idiom)
        assoc_idioms.append(ngram_text)

        for s in row_sample.to_dict(orient="records"):
            matched_ngram = ast.literal_eval(s["ngram"])
            text = " ".join([x.split("/")[0] for x in matched_ngram])
            print(text)
            try:
                left_text, right_text = parse_text(text)
            except:
                pdb.set_trace()
            phrases.append(text)
            left_texts.append(left_text)
            right_texts.append(right_text)
            assoc_idioms.append(ngram_text)
        
        
    out_df = pd.DataFrame({"phrase": phrases, "assoc_idiom": assoc_idioms, "left": left_texts, "right": right_texts})
    groups = out_df["assoc_idiom"].unique()
    random.shuffle(groups)
    out_df = out_df.set_index("assoc_idiom").loc[groups].reset_index()
    out_df.to_csv("./data/human_experiments/sample_for_mturk_match3.csv", index=False)
