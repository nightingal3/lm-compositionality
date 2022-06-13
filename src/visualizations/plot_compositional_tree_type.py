import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import pandas as pd
import numpy as np
from typing import List
from scipy.stats import kruskal
import pdb
import os

import sys
sys.path.append('../ptitprince/')
import ptitprince as pt

colors = {
    "S": "blue",
    "SBAR": "mediumslateblue",
    "NP": "orange",
    "NP-SBJ": "gold",
    "PP": "green",
    "PP-TMP": "lightseagreen",
    "PP-LOC": "turquoise",
    "PP-CLR": "darkcyan",
    "VP": "tomato"
}

def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)

def merge_vecs(arrs: List) -> np.ndarray:
    return np.concatenate(arrs)
    
def filter_rare_trees(merged_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return merged_df.groupby("tree_type").filter(lambda x: len(x) >= threshold)

def record_tree_stats(filtered_df: pd.DataFrame, dist_col: str = "dist", by_parent_type: bool = False) -> pd.DataFrame:
    tree_types = []
    dists = []
    dists_for_test = []
    type_col = "tree_type"
    if by_parent_type:
        filtered_df["parent_type"] = filtered_df["tree_type"].str.split(" -> ", expand=True)[0]
        type_col = "parent_type"

    for tree_type in filtered_df[type_col].unique():
        if "->" not in tree_type and not by_parent_type:
            continue # exclude unary trees
        rows = filtered_df.loc[(filtered_df[type_col] == tree_type) & (filtered_df[dist_col] != -1)]
        if len(rows) == 0:
            continue
        tree_types.extend([tree_type] * len(rows))
        dists_for_test.append(list(rows[dist_col]))
        dists.extend(list(rows[dist_col]))

    cols = {"tree_type": tree_types, "dist": dists}
    print(kruskal(*dists_for_test))
    return pd.DataFrame(cols)

def plot_tree_stats(tree_data: pd.DataFrame, out_filename: str = "tree_dists", dist_col: str = "dist", palette = []) -> None:
    dx = "tree_type"; dy = dist_col; ort = "v"; pal = "Set2"; sigma = .2
    f, ax = plt.subplots(figsize=(7, 5))
    ranks = tree_data.groupby("tree_type")[dist_col].mean().fillna(0).sort_values()[::-1].index
    #palette = [colors[x] for x in ranks]
    #pt.RainCloud(x = dx, y = dy, data = tree_data, palette = pal, bw = sigma,
                    #width_viol = .6, ax = ax, orient = ort, order=order)

    #ax=sns.boxplot( x = dx, y = dy, data = tree_data, zorder = 10,\
            #showcaps = True, showfliers=False, saturation = 1, orient = ort, order=ranks)
    #ax=sns.swarmplot( x = dx, y = dy, data = tree_data, zorder = 10,\
            #orient = ort, order=ranks)
    ax = sns.barplot(x=dx, y=dy, data=tree_data, ci=95, order=ranks, palette=palette)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")

def print_tree_stats(tree_data: pd.DataFrame) -> None:
    tree_types = tree_data["tree_type"].unique()
    data = []
    for t in tree_types:
        rows = tree_data.loc[tree_data["tree_type"] == t]
        data.append((t, rows["dist"].mean()))

    print("Tree types by deviation:")
    data.sort(key=lambda x: x[1])
    for d in data:
        print(f"{d[0]}: {d[1]}")

def get_tree_results(filtered_df: pd.DataFrame, out_filename: str) -> None:
    tree_types = filtered_df["tree_type"].unique()
    for tree_type in tree_types:
        if "->" not in tree_type:
            continue
        selected_cols = filtered_df.loc[filtered_df["tree_type"] == tree_type]
        sorted_vals = selected_cols.sort_values(by="dist_from_children", ascending=False)[["sent", "tree_type", "dist_from_children"]]
        sorted_vals = sorted_vals.drop_duplicates(subset=["sent"])
        sorted_vals.to_csv(os.path.join(out_filename, f"{tree_type}.csv"), index=False)

def plot_comp_score_vs_length(full_df: pd.DataFrame, out_filename: str) -> None:
    full_df = full_df.reset_index()
    full_df["sent_length"] = full_df["sent"].str.split().str.len()
    non_leaves = full_df.loc[full_df["dist_from_children"] >= 0]
    pdb.set_trace()
    sns.lineplot(data=non_leaves, x="sent_length", y="dist_from_children")
    plt.savefig(out_filename)


def plot_deviation_from_output_files(output_df: pd.DataFrame, out_filename: str, by_type: str = "tree") -> None:
    unique_tree_types = output_df["tree_type"].unique()
    output_df["parent_type"] = output_df["tree_type"].str[:2]
    unique_parent_types = output_df["parent_type"].unique()
    iter_categories = unique_tree_types if by_type == "tree" else unique_parent_types

    for tree_type in iter_categories:
        pass

def tree_examples(data: pd.DataFrame, cat: str, out_filename: str) -> None:
    data = data.loc[data["named_ent_types"] == cat]
    data = data.drop_duplicates(subset=["sent"])
    data = data.sort_values(by="cos_score")[["sent", "cos_score"]]
    data.to_csv(out_filename, index=False)

if __name__ == "__main__":
    emb_types = ["avg", "CLS"]
    model_types = ["bert", "roberta", "deberta", "gpt2"]
    palette = None
    for emb_type in emb_types:
        for model_type in model_types:
            df_paths = [f"./data/{model_type}_affine_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
            dfs = [pd.read_csv(path) for path in df_paths]
            merged_df = merge_dataframes(dfs)
            merged_df.to_csv(f"./data/{model_type}_affine_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)

            merged_df = filter_rare_trees(merged_df, 2000)
            tree_stats = record_tree_stats(merged_df, dist_col="deviation", by_parent_type=False)
            if palette == None:
                unique = tree_stats["tree_type"].unique()
                palette = dict(zip(unique, sns.color_palette(palette="rainbow", n_colors=len(unique))))

            plot_tree_stats(tree_stats, f"deviation_{emb_type}_{model_type}_affine", palette=palette)
            print_tree_stats(tree_stats)
