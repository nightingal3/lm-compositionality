import pandas as pd
import pdb
import torch
from typing import List
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from src.models.get_vecs import model_init

bird_datapath = "./data/BiRD/BiRD.txt"

def get_compositionality_corr_bird(model, tokenizer, input_compounds, layers=[12], alpha_add=1, beta_add=1, alpha_mult=1, beta_mult=1, mode="cls") -> List:
    scores = []

    for i in range(len(input_compounds)):
        w1, w2 = input_compounds.iloc[i]["term1"], input_compounds.iloc[i]["term2"]
        
        print("w1:", w1, "w2:", w2)
    
        encoded_w1 = tokenizer(w1, padding=True, truncation=True, return_tensors="pt")
        encoded_w2 = tokenizer(w2, padding=True, truncation=True, return_tensors="pt")

        outputs_w1 = model(encoded_w1["input_ids"], token_type_ids=None, attention_mask=encoded_w1["attention_mask"])
        outputs_w2 = model(encoded_w2["input_ids"], token_type_ids=None, attention_mask=encoded_w2["attention_mask"])

        selected_layers_w1 = torch.stack([outputs_w1["hidden_states"][l].squeeze(0) for l in layers]) # each layer (num layers x batch size (3) x length x hidden size)
        selected_layers_w2 = torch.stack([outputs_w2["hidden_states"][l].squeeze(0) for l in layers]) # each layer (num layers x batch size (3) x length x hidden size)

        if mode == "cls": #TODO: handle layers properly
            v1 = selected_layers_w1[0][0]
            v2 = selected_layers_w2[0][0]
        elif mode == "mean-all":
            v1 = selected_layers_w1.squeeze(0).mean(dim=0)
            v2 = selected_layers_w2.squeeze(0).mean(dim=0)
        elif mode == "mean-words":
            v1 = selected_layers_w1.squeeze(0)[1:-1].mean(dim=0)
            v2 = selected_layers_w2.squeeze(0)[1:-1].mean(dim=0)
        elif mode == "max":
            v1 = selected_layers_w1.squeeze(0).max(dim=0)
            v2 = selected_layers_w2.squeeze(0).max(dim=0)
        elif mode == "sep":
            v1 = selected_layers_w1[0][-1]
            v2 = selected_layers_w2[0][-1]
        else:
            raise NotImplementedError

        scores.append(torch.nn.functional.cosine_similarity(v1, v2, dim=0).item())
        

    return scores

def get_correlation_across_layers_bird(input_data: pd.DataFrame, comparison_data: pd.Series, out_filename: str = "layer_corr.png", model_name: str = "bert", single_layers: bool = True) -> None:
    layer_range = [1, 13]
    
    model, tokenizer = model_init(model_name, False)

    layers = [i for i in range(layer_range[0], layer_range[1])]
    corrs = {"cls": [], "avg-phrase": [], "avg-all": []}
    curr_layers = []

    for layer in layers:
        if not single_layers:
            curr_layers.append(layer)
        else:
            curr_layers = [layer]
        sim_cls = get_compositionality_corr_bird(model, tokenizer, input_data, mode="cls",layers=curr_layers)
        sim_avg = get_compositionality_corr_bird(model, tokenizer, input_data, mode="mean-all", layers=curr_layers)
        sim_avg_words = get_compositionality_corr_bird(model, tokenizer, input_data, mode="mean-words", layers=curr_layers)

        corrs["cls"].append(get_bert_correlation_human_judgment(sim_cls, comparison_data)[0])
        corrs["avg-phrase"].append(get_bert_correlation_human_judgment(sim_avg_words, comparison_data)[0])
        corrs["avg-all"].append(get_bert_correlation_human_judgment(sim_avg, comparison_data)[0])

    ax = plt.gca()
    plt.plot(layers, corrs["cls"], label="CLS")
    plt.plot(layers, corrs["avg-all"], label="AVG")
    plt.plot(layers, corrs["avg-phrase"], label="AVG-PHRASE")
    plt.xlabel("Layer")
    plt.ylabel("Spearman corr.")
    plt.legend()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")

def get_bert_correlation_human_judgment(bert_scores, human_scores) -> float:
    return spearmanr(bert_scores, list(human_scores))

if __name__ == "__main__":
    df = pd.read_csv(bird_datapath, sep="\t")
    compounds = df[["term1", "term2"]]
    relatedness = df["relatedness score"]
    get_correlation_across_layers_bird(compounds, relatedness, model_name="bert", out_filename="bert-corr-bird")
    get_correlation_across_layers_bird(compounds, relatedness, model_name="roberta", out_filename="roberta-corr-bird")
