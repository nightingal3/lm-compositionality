import pandas as pd
from typing import Callable
from scipy.spatial.distance import cosine
import pdb
import ast
from typing import List
import argparse

from composition_functions import *


# Gets the cosine distance between a node and some composition of its children.
def compare_node_to_children(tree_data: pd.DataFrame, vecs: np.ndarray, composition_fn: Callable, add_proto_embs: bool = False) -> List:
    cosine_distance_to_children = []
    for i, row in tree_data.iterrows():
        print(i)
        #full_sent, parent_emb, tree_pos, depth = row[1], ast.literal_eval(row[8]), row[6], row[5]
        full_sent, parent_emb, tree_pos, depth = row[1], vecs[i], row[6], row[5]
        #child_embs = get_child_embs(tree_data, full_sent, tree_pos, depth)
        child_embs = get_child_embs_separate(tree_data, full_sent, tree_pos, depth, vecs)
        if len(child_embs) == 0 or len(child_embs) == 1:
            cosine_distance_to_children.append(-1) # leaf node, or unary, mark with -1 to indicate no children
        else:
            if add_proto_embs:
                proto_emb = ast.literal_eval(row[8])
                child_embs.append(proto_emb)
                
            child_comp_emb = composition_fn(child_embs)
            cosine_distance_to_children.append(cosine(child_comp_emb, parent_emb))

    return cosine_distance_to_children

def get_child_embs(tree_data: pd.DataFrame, full_sent: str, tree_pos: tuple, depth: int) -> np.ndarray:
    candidates = tree_data.loc[tree_data["full_sent"] == full_sent]
    if tree_pos == "ROOT":
        children = np.array([ast.literal_eval(emb) for emb in candidates.loc[candidates["depth"] == 1]["emb"]])
    else:
        right_depth = candidates.loc[candidates["depth"] == depth + 1]            
        children = [ast.literal_eval(emb) for emb in right_depth.loc[right_depth["tree_ind"].str.startswith(tree_pos[:-1])]["emb"]]
        children = np.array(children)
    return children
    
def get_child_embs_separate(tree_data: pd.DataFrame, full_sent: str, tree_pos: tuple, depth: int, vecs: np.ndarray) -> np.ndarray:
    candidates = tree_data.loc[tree_data["full_sent"] == full_sent]
    if tree_pos == "ROOT":
        child_inds = candidates[candidates["depth"] == 1].index.tolist()
        children = vecs[child_inds, :]
    else:
        right_depth = candidates.loc[candidates["depth"] == depth + 1]            
        child_inds = right_depth[right_depth["tree_ind"].str.startswith(tree_pos[:-1])].index.tolist()
        children = vecs[child_inds, :]
    return children

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("-i", dest="selection_num", help="process number (split into 10)")
    parser.add_argument("--emb_type", help="embedding type to use", choices=["CLS", "avg", "max"], default="CLS")
    parser.add_argument("--full", help="use all treebank data", action="store_true")
    parser.add_argument("--cuda", help="use gpu", action="store_true")

    args = parser.parse_args()
    pdb.set_trace()
    if args.cuda:
        import cupy as np
    else:
        import numpy as np

    i, emb_type, treebank_full = args.selection_num, args.emb_type, args.full
    if treebank_full:
        df = pd.read_csv(f"./tree_data_{i}_{emb_type}_{treebank_full}.csv")
        vecs = np.load(f"./data/embs/{emb_type}_{i}_full_True.npy")
    else:
        df = pd.read_csv(f"./tree_data_{i}_{emb_type}.csv")
        vecs = np.load(f"./data/embs/{emb_type}_{i}.npy")

    dist_to_children = compare_node_to_children(df, vecs, add)
    df["dist_from_children"] = dist_to_children
    df.to_csv(f"./data/tree_data_{i}_{emb_type}_full_{treebank_full}.csv")