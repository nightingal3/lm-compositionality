import pandas as pd
import pdb
from typing import List
import ast
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import pickle
from pylab import figure
from scipy.spatial.distance import cdist

def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)

def get_prototype_vec(df: pd.DataFrame, emb_type: str = "mean") -> np.ndarray:
    embs = np.array(df["emb"].apply(lambda x: np.array(ast.literal_eval(x))).tolist())
    
    if emb_type == "mean":
        proto = np.einsum('ij->j', embs)/len(embs)
    else:
        raise NotImplementedError
    
    return proto

def get_pos_prototypes(df: pd.DataFrame, threshold: int = 100) -> dict:
    cat_prototypes = {}
    print("getting prototypes...")
    for tree_type in df["tree_type"].unique():
        selected_rows = df.loc[df["tree_type"] == tree_type]
        if len(selected_rows) < threshold:
            continue
        print(tree_type)
        proto = get_prototype_vec(selected_rows)
        cat_prototypes[tree_type] = proto
    return cat_prototypes

def plot_cat_prototypes(proto_data: dict, out_filename: str) -> None:
    embs = np.array([emb for tree_type, emb in proto_data.items()])
    labels = [tree_type for tree_type, emb in proto_data.items()]
    pca = PCA(n_components=3, whiten=True)
    pca.fit(embs)
    emb_pca = pca.transform(embs)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(labels))))
    fig = figure()
    ax = fig.add_subplot(projection="3d")

    for i, (tree_type, emb) in enumerate(proto_data.items()):
        ax.scatter(emb_pca[i, 0], emb_pca[i, 2], emb_pca[i, 1],
            color=next(colors))
        ax.text(emb_pca[i, 0], emb_pca[i, 2], emb_pca[i, 1], labels[i], size=8)
    #plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
    plt.savefig(out_filename)

def vector_analogies(cat_prototypes: dict) -> None:
    compositions = {tree_type for tree_type in cat_prototypes if "->" in tree_type}
    embs = np.array([emb for tree_type, emb in cat_prototypes.items()])
    labels = [tree_type for tree_type, emb in cat_prototypes.items()]
    neighbours = NearestNeighbors(n_neighbors=3, metric='cosine')
    neighbours.fit(embs)

    for comp_type in compositions: 
        print(comp_type)
        children = comp_type.split("->")[1].split()
        root = comp_type.split("->")[0][:-1]
        try:
            children_vecs = np.array([cat_prototypes[child] for child in children])
        except:
            print("child not in labels")
            continue
        sum_vec = np.einsum("ij->j", children_vecs)
        knn_emb, knn_inds = neighbours.kneighbors(sum_vec.reshape(1, -1), 3)
        print("closest: ")
        if root not in labels:
            print("note: root not in labels")
        for ind in knn_inds[0]:
            print(labels[ind])

def dist_to_prototypes(df: pd.DataFrame, vecs: np.ndarray, cat_prototypes:dict) -> pd.DataFrame:
    new_df = df.copy(deep=True)
    all_dists = np.zeros((len(df), 1))
    proto_vecs = np.zeros((len(df), 768))
    mask = np.ones((len(df), 1), dtype=bool)
    print("finding dist to prototypes...")
    for tree_type in cat_prototypes:
        print(tree_type)
        selected_rows = df.loc[df["tree_type"] == tree_type]
        selected_vecs = np.array([vecs[i] for i in selected_rows.index])
        proto_matrix = np.tile(cat_prototypes[tree_type], (len(selected_vecs), 1))
        dists = [x[0] for x in cdist(selected_vecs, proto_matrix, metric="cosine")]
        for i, real_ind in zip(range(len(dists)), selected_rows.index):
            all_dists[real_ind] = dists[i]
            mask[real_ind] = False
            proto_vecs[real_ind] = cat_prototypes[tree_type]
    
    all_dists[mask] = -1
    proto_vecs = [x.tolist() for x in proto_vecs]
    new_df["proto_emb"] = proto_vecs
    new_df["dist_from_proto"] = all_dists
    return new_df

def parent_distance_child_and_prototype(df: pd.DataFrame, cat_prototypes: dict, vecs: np.ndarray) -> pd.DataFrame:
    pass

if __name__ == "__main__":
    emb_type = "CLS"
    df_paths = [f"./data/tree_data_{i}_{emb_type}_full_True.csv" for i in range(0, 10)]
    dfs = [pd.read_csv(path) for path in df_paths]
    df = merge_dataframes(dfs)
    vecs = np.load(f"./data/embs/CLS_all_full.npy")

    #cat_prototypes = get_pos_prototypes(df)
    cat_prototypes = pickle.load(open("cat_proto_full.p", "rb"))
    #pickle.dump(cat_prototypes, open("cat_proto_full.p", "wb"))
    #vector_analogies(cat_prototypes)
    #plot_cat_prototypes(cat_prototypes, "pca_tree_type.png")
    new_df = dist_to_prototypes(df, vecs, cat_prototypes)
    new_df.to_csv("./proto-dist-full.csv", index=False)

