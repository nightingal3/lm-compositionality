import pandas as pd
import pdb
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List
import re
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

compounds_path = "./data/ijcnlp_compositionality_data/MeanAndDeviations.clean.txt"

def get_compositionality_scores_bert(model, tokenizer, input_compounds, alpha_add=1, beta_add=1, alpha_mult=1, beta_mult=1) -> List:
    add_scores = []
    mult_scores = []
    w1_scores = []
    w2_scores = []

    for i in range(len(input_compounds)):
        compound, w1, w2 = input_compounds.iloc[i]["compound_phrase"], input_compounds.iloc[i]["index"], input_compounds.iloc[i]["#word"]
        compound, w1, w2 = re.sub(r"-[a-z]", "", compound), re.sub(r"-[a-z]", "", w1), re.sub(r"-[a-z]", "", w2)
        
        encoded_inputs = tokenizer([compound, w1, w2], padding=True, truncation=True, return_tensors="pt")
        outputs = model(encoded_inputs["input_ids"], token_type_ids=None, attention_mask=encoded_inputs["attention_mask"])
        compound_vec, v1, v2 = [outputs["last_hidden_state"][i][0].detach().numpy() for i in range(3)]
        add_scores.append(cosine(compound_vec, v1 + v2))
        mult_scores.append(cosine(compound_vec, v1 * v2))
        w1_scores.append(cosine(compound_vec, v1))
        w2_scores.append(cosine(compound_vec, v2))

    return add_scores, mult_scores, w1_scores, w2_scores

def get_compositionality_scores_flair():
    raise NotImplementedError


def get_bert_correlation_human_judgment(bert_scores, human_scores) -> float:
    return pearsonr(bert_scores, list(human_scores))

if __name__ == "__main__":
    columns = ['#word', 'Word1_mean', 'Word1_std', 'Word2_mean', 'Word2_std', 'Cpd_mean', 'Cpd_std', 'mean1*mean2']
    compounds_df = pd.read_csv(compounds_path, delimiter="\t|\s+", names=columns, header=None, skiprows=1)
    compounds_df = compounds_df.reset_index(level=0)
    compounds_df["compound_phrase"] = compounds_df["index"] + " " + compounds_df["#word"]
    input_data, comp_human_judgements = compounds_df[["compound_phrase", "index", "#word"]], compounds_df["Cpd_mean"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained('bert-base-uncased',
    output_attentions = False, 
    output_hidden_states = True)

    add_scores, mult_scores, w1_scores, w2_scores = get_compositionality_scores_bert(model, tokenizer, input_data)
    print("ADD: ", get_bert_correlation_human_judgment(add_scores, comp_human_judgements))
    print("MULT: ", get_bert_correlation_human_judgment(mult_scores, comp_human_judgements))
    print("W1: ", get_bert_correlation_human_judgment(w1_scores, comp_human_judgements))
    print("W2: ", get_bert_correlation_human_judgment(w2_scores, comp_human_judgements))

