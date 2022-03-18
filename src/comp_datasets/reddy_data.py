import argparse
import pandas as pd
import pdb
import numpy as np
from transformers import BertTokenizer, BertModel, DebertaTokenizer, DebertaModel
from typing import List
import re
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.models.get_vecs import model_init

compounds_path = "./data/ijcnlp_compositionality_data/MeanAndDeviations.clean.txt"

def get_compositionality_scores(model, tokenizer, input_compounds, layers=[12], alpha_add=1, beta_add=1, alpha_mult=1, beta_mult=1, mode="cls") -> List:
    add_scores = []
    mult_scores = []
    w1_scores = []
    w2_scores = []

    for i in range(len(input_compounds)):
        compound, w1, w2 = input_compounds.iloc[i]["compound_phrase"], input_compounds.iloc[i]["index"], input_compounds.iloc[i]["#word"]
        compound, w1, w2 = re.sub(r"-[a-z]", "", compound), re.sub(r"-[a-z]", "", w1), re.sub(r"-[a-z]", "", w2)
        
        print(compound)
        print("w1:", w1, "w2:", w2)
    
        encoded_compound = tokenizer(compound, padding=True, truncation=True, return_tensors="pt")
        encoded_w1 = tokenizer(w1, padding=True, truncation=True, return_tensors="pt")
        encoded_w2 = tokenizer(w2, padding=True, truncation=True, return_tensors="pt")

        outputs_compound = model(encoded_compound["input_ids"], token_type_ids=None, attention_mask=encoded_compound["attention_mask"])
        outputs_w1 = model(encoded_w1["input_ids"], token_type_ids=None, attention_mask=encoded_w1["attention_mask"])
        outputs_w2 = model(encoded_w2["input_ids"], token_type_ids=None, attention_mask=encoded_w2["attention_mask"])

        selected_layers_compound = torch.stack([outputs_compound["hidden_states"][l].squeeze(0) for l in layers]) # each layer (num layers x seq length x hidden size)
        selected_layers_w1 = torch.stack([outputs_w1["hidden_states"][l].squeeze(0) for l in layers]) # each layer (num layers x seq length x hidden size)
        selected_layers_w2 = torch.stack([outputs_w2["hidden_states"][l].squeeze(0) for l in layers]) # each layer (num layers x seq length x hidden size)

        if mode == "cls": #TODO: handle layers properly
            compound_vec = selected_layers_compound[0][0]
            v1 = selected_layers_w1[0][0]
            v2 = selected_layers_w2[0][0]
        elif mode == "mean-all":
            compound_vec = selected_layers_compound.squeeze(0).mean(dim=0)
            v1 = selected_layers_w1.squeeze(0).mean(dim=0)
            v2 = selected_layers_w2.squeeze(0).mean(dim=0)
        elif mode == "mean-words":
            compound_vec = selected_layers_compound.squeeze(0)[1:-1].mean(dim=0)
            v1 = selected_layers_w1.squeeze(0)[1:-1].mean(dim=0)
            v2 = selected_layers_w2.squeeze(0)[1:-1].mean(dim=0)
        elif mode == "sep":
            compound_vec = selected_layers_compound[0][-1]
            v1 = selected_layers_w1[0][-1]
            v2 = selected_layers_w2[0][-1]
        elif mode == "max":
            compound_vec = selected_layers_compound.squeeze(0).max(dim=0)
            v1 = selected_layers_w1.squeeze(0).max(dim=0)
            v2 = selected_layers_w2.squeeze(0).max(dim=0)
        else:
            raise NotImplementedError

        add_scores.append(torch.nn.functional.cosine_similarity(compound_vec, v1 + v2, dim=0).item())
        mult_scores.append(torch.nn.functional.cosine_similarity(compound_vec, v1 * v2, dim=0).item())
        w1_scores.append(torch.nn.functional.cosine_similarity(compound_vec, v1, dim=0).item())
        w2_scores.append(torch.nn.functional.cosine_similarity(compound_vec, v2, dim=0).item())

    return add_scores, mult_scores, w1_scores, w2_scores

def get_compositionality_scores_flair(embeddings, word_embeddings, input_compounds, layers: str = "all", word_embed_type: str = "doc", model_type: str = "bert-base-uncased", wrong: bool = False):
    add_scores = []
    mult_scores = []
    w1_scores = []
    w2_scores = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(len(input_compounds)):
        compound, w1, w2 = input_compounds.iloc[i]["compound_phrase"], input_compounds.iloc[i]["index"], input_compounds.iloc[i]["#word"]
        compound, w1, w2 = re.sub(r"-[a-z]", "", compound), re.sub(r"-[a-z]", "", w1), re.sub(r"-[a-z]", "", w2)

        print(compound)

        # use only last layers
        sent = Sentence(compound)

        if wrong:
            word_embeddings.embed(sent)
            compound_vec = sent[0].embedding
        else:
            embeddings.embed(sent)
            compound_vec = sent.embedding

        w1_sent = Sentence(w1)
        if word_embed_type == "doc":
            embeddings.embed(w1_sent)
            v1 = w1_sent.embedding
        else:
            word_embeddings.embed(w1_sent)
            v1 = w1_sent[0].embedding

        w2_sent = Sentence(w2)
        if word_embed_type == "doc":
            embeddings.embed(w2_sent)
            v2 = w2_sent.embedding
        else:
            word_embeddings.embed(w2_sent)
            v2 = w2_sent[0].embedding

        print(cos(compound_vec, v1 + v2).item())
        add_scores.append(cos(compound_vec, v1 + v2).item())
        mult_scores.append(cos(compound_vec, v1 * v2).item())
        w1_scores.append(cos(compound_vec, v1).item())
        w2_scores.append(cos(compound_vec, v2).item())

    return add_scores, mult_scores, w1_scores, w2_scores

def get_bert_correlation_human_judgment(bert_scores, human_scores) -> float:
    return spearmanr(bert_scores, list(human_scores))


def get_correlation_across_layers_flair(input_data: pd.DataFrame, comparison_data: pd.Series, out_filename: str = "layer_corr.png", model_name: str = "bert-base-uncased", single_layers: bool = False, word_embed_type: str = "word", plot_layers: bool = True) -> None:
    layer_range = [1, 13]

    layers = [-i for i in range(layer_range[0], layer_range[1])]
    true_layers = [i for i in reversed(range(layer_range[0], layer_range[1]))]
    corrs = {"add": [], "mult": [], "w1": [], "w2": []}
    curr_layers = ""

    for layer in layers:
        if single_layers:
            curr_layers = f"{layer}"
        else:
            if layer == -1:
                curr_layers += "-1"
            else:
                curr_layers += f",{layer}"
        embeddings = TransformerDocumentEmbeddings(model_name, layers=curr_layers, cls_pooling="cls")
        word_embeddings = TransformerWordEmbeddings(model_name, layers=curr_layers, layer_mean=False)

        add_scores, mult_scores, w1_scores, w2_scores = get_compositionality_scores_flair(embeddings, word_embeddings, input_data, mode="cls", layers=curr_layers)
        corrs["add"].append(get_bert_correlation_human_judgment(add_scores, comparison_data)[0])
        corrs["mult"].append(get_bert_correlation_human_judgment(mult_scores, comparison_data)[0])
        corrs["w1"].append(get_bert_correlation_human_judgment(w1_scores, comparison_data)[0])
        corrs["w2"].append(get_bert_correlation_human_judgment(w2_scores, comparison_data)[0])


    ax = plt.gca()
    plt.plot(true_layers, corrs["add"], label="ADD")
    plt.plot(true_layers, corrs["mult"], label="MULT")
    plt.plot(true_layers, corrs["w1"], label="W1")
    plt.plot(true_layers, corrs["w2"], label="W2")
    ax.invert_xaxis()
    plt.xlabel("Layer")
    plt.ylabel("Spearman corr.")
    plt.legend()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")


def get_correlation_across_layers(input_data: pd.DataFrame, comparison_data: pd.Series, out_filename: str = "layer_corr.png", model_name: str = "bert-base-uncased", single_layers: bool = True, word_embed_type: str = "cls", plot_layers: bool = True) -> None:
    layer_range = [1, 13]
    
    model, tokenizer = model_init(model_name, False)

    layers = [i for i in reversed(range(layer_range[0], layer_range[1]))]
    corrs = {"add": [], "mult": [], "w1": [], "w2": []}
    all_sim = {"add": [], "mult": [], "w1": [], "w2": []}

    
    curr_layers = []

    for layer in layers:
        if not single_layers:
            curr_layers.append(layer)
        else:
            curr_layers = [layer]
        add_scores, mult_scores, w1_scores, w2_scores = get_compositionality_scores(model, tokenizer, input_data, mode=word_embed_type,layers=curr_layers)
        corrs["add"].append(get_bert_correlation_human_judgment(add_scores, comparison_data)[0])
        corrs["mult"].append(get_bert_correlation_human_judgment(mult_scores, comparison_data)[0])
        corrs["w1"].append(get_bert_correlation_human_judgment(w1_scores, comparison_data)[0])
        corrs["w2"].append(get_bert_correlation_human_judgment(w2_scores, comparison_data)[0])

        all_sim["add"].append(add_scores)
        all_sim["mult"].append(mult_scores)
        all_sim["w1"].append(w1_scores)
        all_sim["w2"].append(w2_scores)

    ax = plt.gca()
    plt.plot(layers, corrs["add"], label="ADD")
    plt.plot(layers, corrs["mult"], label="MULT")
    plt.plot(layers, corrs["w1"], label="W1")
    plt.plot(layers, corrs["w2"], label="W2")
    ax.invert_xaxis()
    plt.xlabel("Layer")
    plt.ylabel("Spearman corr.")
    plt.legend()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")

    if plot_layers:
        for i in range(len(layers)):
            plt.gcf().clear()
            plt.hist(all_sim["add"][i], label=f"ADD ({layers[i]})", alpha=0.4)
            print(layers[i])
            print(np.array(all_sim["add"][i]).mean())
            print(np.array(all_sim["add"][i]).std())

            plt.hist(all_sim["mult"][i], label=f"MULT ({layers[i]})", alpha=0.4)
            plt.hist(all_sim["w1"][i], label=f"W1 ({layers[i]})", alpha=0.4)
            plt.hist(all_sim["w2"][i], label=f"W2 ({layers[i]})", alpha=0.4)
            plt.xlabel("Cosine similarity")
            plt.legend()
            plt.savefig(f"layer_cos_sim_{model_name}_{word_embed_type}_{layers[i]}.png")
            plt.savefig(f"layer_cos_sim_{model_name}_{word_embed_type}_{layers[i]}.eps")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("model_name", help="model to examine", choices=["bert", "roberta", "deberta"])
    parser.add_argument("--individual", help="whether to examine correlation with individual or cumulative layers. Defaults to cumulative.", action="store_true")
    parser.add_argument("--out_filename", help="filename for generated figure", default="layer_corr")
    parser.add_argument("--cls", help="compare CLS tokens of single words (default is word embeddings)", action="store_true")
    parser.add_argument("--freq", help="compare TRE with frequency data rather than human compositionality judgments", choices=["abs", "rel"])
    parser.add_argument("--flair", help="use flair vs. manual", action="store_true")
    parser.add_argument("--type", dest="word_embed_type", choices=["cls", "mean-words", "mean-all", "sep"], default="cls")
    args = parser.parse_args()

    columns = ['#word', 'Word1_mean', 'Word1_std', 'Word2_mean', 'Word2_std', 'Cpd_mean', 'Cpd_std', 'mean1*mean2']
    compounds_df = pd.read_csv(compounds_path, delimiter="\t|\s+", names=columns, header=None, skiprows=1)
    compounds_df = compounds_df.reset_index(level=0)
    compounds_df["compound_phrase"] = compounds_df["index"] + " " + compounds_df["#word"]
    input_data, comp_human_judgments = compounds_df[["compound_phrase", "index", "#word"]], compounds_df["Cpd_mean"]

    freq_df = pd.read_csv("./data/ijcnlp_compositionality_data/freq.csv")
    freq_data = freq_df["freq"]
    relative_freq_data = (freq_df["w1_freq"] + freq_df["w2_freq"]) - freq_df["freq"]

    model_dict = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-base"
    }
    model_name = model_dict[args.model_name]
    word_embed_type = "doc" if args.cls else "word"

    if args.freq:
        comparison_data = freq_data if args.freq == "abs" else relative_freq_data
    else:
        comparison_data = comp_human_judgments
    
    if args.flair:
        get_correlation_across_layers_flair(input_data, comparison_data, model_name=model_name, single_layers=args.individual, out_filename=args.out_filename, word_embed_type=word_embed_type)
    else:
        get_correlation_across_layers(input_data, comparison_data, model_name=args.model_name, out_filename=args.out_filename, single_layers=args.individual, word_embed_type=args.word_embed_type)

