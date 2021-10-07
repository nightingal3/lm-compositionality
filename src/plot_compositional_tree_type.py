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

def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs)

def merge_vecs(arrs: List) -> np.ndarray:
    return np.concatenate(arrs)
    
def filter_rare_trees(merged_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return merged_df.groupby("tree_type").filter(lambda x: len(x) >= threshold)

def record_tree_stats(filtered_df: pd.DataFrame) -> pd.DataFrame:
    tree_types = []
    dists = []
    dists_for_test = []
    for tree_type in filtered_df["tree_type"].unique():
        if "->" not in tree_type:
            continue # exclude unary trees
        rows = filtered_df.loc[(filtered_df["tree_type"] == tree_type) & (filtered_df["dist_from_children"] != -1)]
        if len(rows) == 0:
            continue
        tree_types.extend([tree_type] * len(rows))
        dists_for_test.append(list(rows["dist_from_children"]))
        dists.extend(list(rows["dist_from_children"]))

    cols = {"tree_type": tree_types, "dist": dists}
    pdb.set_trace()
    print(kruskal(*dists_for_test))
    return pd.DataFrame(cols)

def plot_tree_stats(tree_data: pd.DataFrame, out_filename: str = "tree_dists.png") -> None:
    dx = "tree_type"; dy = "dist"; ort = "v"; pal = "Set2"; sigma = .2
    f, ax = plt.subplots(figsize=(7, 5))

    pt.RainCloud(x = dx, y = dy, data = tree_data, palette = pal, bw = sigma,
                    width_viol = .6, ax = ax, orient = ort)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_filename)

def get_tree_results(filtered_df: pd.DataFrame, out_filename: str) -> None:
    tree_types = filtered_df["tree_type"].unique()
    for tree_type in tree_types:
        if "->" not in tree_type:
            continue
        selected_cols = filtered_df.loc[filtered_df["tree_type"] == tree_type]
        sorted_vals = selected_cols.sort_values(by="dist_from_children", ascending=False)[["sent", "tree_type", "dist_from_children"]]
        sorted_vals = sorted_vals.drop_duplicates(subset=["sent"])
        sorted_vals.to_csv(os.path.join(out_filename, f"{tree_type}.csv"), index=False)

if __name__ == "__main__":
    emb_type = "avg"
    threshold = 500
    df_paths = [f"./data/tree_data_{i}_{emb_type}.csv" for i in range(0, 10)]
    emb_lst = [np.load(f"./data/embs/{emb_type}_{i}.npy") for i in range(0, 10)]
    dfs = [pd.read_csv(path) for path in df_paths]
    merged_df = merge_dataframes(dfs)[['full_length', 'full_sent', 'sent',
       'sublength', 'depth', 'tree_ind', 'tree_type', 'emb', 'dist_from_root',
       'dist_from_children']]
    merged_df.to_csv(f"./data/{emb_type}_all.csv", index=False)
    df_filtered = filter_rare_trees(merged_df, threshold)
    get_tree_results(df_filtered, f"./data/samples/{emb_type}")
    df_filtered.loc[df_filtered["dist_from_children"] >= 0.3].sort_values("dist_from_children", ascending=False)[["sent", "tree_type", "dist_from_children"]].to_csv(f"./high-dist-{emb_type}.csv")
    tree_stats = record_tree_stats(df_filtered)
    plot_tree_stats(tree_stats, f"tree_dists_{emb_type}.png")
