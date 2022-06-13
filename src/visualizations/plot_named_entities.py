import seaborn as sns
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from pathlib import Path
import argparse
import joypy
from matplotlib import cm

from src.visualizations.plot_compositional_tree_type import merge_dataframes
from src.models.fit_composition_functions import AffineRegression

def plot_cos_dist_hist(df: pd.DataFrame, out_filename: str, dim: int = 1, plot_type: str = "hist") -> None:
    if plot_type == "hist":
        if dim == 1:
            sns.histplot(data=df, x="cos_score", hue="is_named_entity")
        else:
            sns.histplot(data=df, x="sublength", y="cos_score", hue="is_named_entity", kde=True)
    else:
        if dim == 1:
            sns.kdeplot(data=df, x="cos_score", hue="is_named_entity", common_norm=False, common_grid=True, fill=True, alpha=0.5)
        else:
            sns.kdeplot(data=df, x="sublength", y="cos_score", hue="is_named_entity", common_norm=False)

    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.pdf")
    plt.savefig(f"{out_filename}.eps")

def plot_named_entity_types(named_entities: pd.DataFrame, out_filename: str, dim: int = 1, plot_type: str = "hist") -> None:
    if plot_type == "hist":
        if dim == 1:
            sns.histplot(data=named_entities, x="cos_score", hue="named_ent_types", kde=True)
        else:
            sns.histplot(data=named_entities, x="sublength", y="cos_score", hue="named_ent_types", kde=True)
    else:
        if dim == 1:
            sns.kdeplot(data=named_entities, x="cos_score", hue="named_ent_types", common_norm=False)
        else:
            sns.kdeplot(data=named_entities, x="sublength", y="cos_score", hue="named_ent_types", common_norm=False)

    plt.tight_layout()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.pdf")
    plt.savefig(f"{out_filename}.eps")

def plot_named_entity_types_joy(named_entities: pd.DataFrame, out_filename: str) -> None:
    fig, ax = joypy.joyplot(named_entities, by="named_ent_types", column="cos_score", colormap=cm.viridis)
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.pdf")
    plt.savefig(f"{out_filename}.eps")

def filter_rare_entities(named_entities: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return named_entities.groupby("named_ent_types").filter(lambda x: len(x) >= threshold)

def glm_model(data: pd.DataFrame) -> None:
    md = smf.mixedlm("cos_score ~ sublength + cos_score", data, groups=data["sublength"], re_formula="~cos_score")
    mdf = md.fit(method=["lbfgs"])
    print(mdf.summary())

def entity_examples(data: pd.DataFrame, cat: str, out_filename: str) -> None:
    data = data.loc[data["named_ent_types"] == cat]
    data = data.drop_duplicates(subset=["sent"])
    data = data.sort_values(by="cos_score")[["sent", "cos_score"]]
    data.to_csv(out_filename, index=False)

def get_dist_from_child_pred(model_type: str, emb_type: str, parent_embs: np.ndarray, child_embs: np.ndarray, df: pd.DataFrame):
    if emb_type == "CLS": #these probably should have been the same dimension to start with...oh well
        sim = torch.nn.CosineSimilarity(dim=1)
    else:
        sim = torch.nn.CosineSimilarity(dim=0)
    composition_model = AffineRegression(input_size=2, output_size=768, cuda=torch.cuda.is_available())
    composition_model.load_state_dict(torch.load(f"./data/{model_type}_affine_{emb_type}_results/binary_trees/full/model_0_bintree.pt")) 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    similarities = []   
    for i in range(parent_embs.shape[0]):
        if df.iloc[i]["binary_text"] == '[]':
            similarities.append(-1)
            continue
        actual_emb = torch.tensor(parent_embs[i]).float().to(device)
        curr_child_embs = torch.tensor(child_embs[i]).float().to(device)
        composed_emb = composition_model(curr_child_embs).squeeze(0)
        cos_sim = sim(actual_emb, composed_emb)
        similarities.append(cos_sim.item())
    
    return similarities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate named entity figures.")
    parser.add_argument("--emb_type", default="CLS", help="embedding type", choices=["CLS", "avg"])
    parser.add_argument("--model", default="bert", help="model type", choices=["bert", "roberta", "deberta", "gpt2"])
    
    args = parser.parse_args()
    emb_type = args.emb_type
    model_type = args.model
    SCRATCH_DIR = "/compute/tir-0-19/mengyan3"

    df = pd.read_csv(f"./data/{model_type}_affine_{emb_type}_results/binary_trees/full/deviations_all.csv")
    df["cos_score"] = 1 - df["deviation"]
    df = df.drop_duplicates(subset=["sent"])

    df_orig = pd.read_csv(f"{SCRATCH_DIR}/large_test/{model_type}_{emb_type}_all_full.csv")
    df = df.merge(df_orig, how="inner", on="sent")

    named_entities = df.loc[(df["is_named_entity"] == True) & (df["cos_score"] != -1)]
    non_named_entities = df.loc[(df["is_named_entity"] == False) & (df["cos_score"] != -1)]
    plot_cos_dist_hist(df, f"ner_test_{model_type}_{emb_type}_full", plot_type="kde")
    plt.gcf().clear()
    named_entities = filter_rare_entities(named_entities, 20)
    #plot_named_entity_types(named_entities, f"ner_entities_{model_type}_{emb_type}_full", plot_type="kde")
    plot_named_entity_types_joy(named_entities, f"ner_joy_{model_type}_{emb_type}")
    #glm_model(df)

    Path(f"./data/examples/{model_type}/{emb_type}").mkdir(parents=True, exist_ok=True)
    entity_examples(named_entities, "['PERSON']", f"./data/examples/{model_type}/{emb_type}/person_examples.csv")
    entity_examples(named_entities, "['DATE']", f"./data/examples/{model_type}/{emb_type}/date_examples.csv")
    entity_examples(named_entities, "['GPE']", f"./data/examples/{model_type}/{emb_type}/gpe_examples.csv")
    entity_examples(named_entities, "['MONEY']", f"./data/examples/{model_type}/{emb_type}/money_examples.csv")
    entity_examples(named_entities, "['PERCENT']", f"./data/examples/{model_type}/{emb_type}/percent_examples.csv")
    entity_examples(named_entities, "['ORG']", f"./data/examples/{model_type}/{emb_type}/org_examples.csv")
    entity_examples(named_entities, "['CARDINAL']", f"./data/examples/{model_type}/{emb_type}/cardinal_examples.csv")
    entity_examples(named_entities, "['QUANTITY']", f"./data/examples/{model_type}/{emb_type}/quantity_examples.csv")
    entity_examples(named_entities, "['LOC']", f"./data/examples/{model_type}/{emb_type}/loc_examples.csv")




