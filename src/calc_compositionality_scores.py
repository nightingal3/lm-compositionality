import pandas as pd
from typing import Callable
from scipy.spatial.distance import cosine
import pdb
import ast
from typing import List
import argparse
from nltk import Tree

from src.composition_functions import *
from src.models.get_vecs import get_one_vec, model_init


# Gets the cosine distance between a node and some composition of its children.
def compare_node_to_children(tree_data: pd.DataFrame, vecs: np.ndarray, composition_fn: Callable, add_proto_embs: bool = False, vec_type: str = "cls") -> List:
    lm_model, lm_tokenizer = model_init("bert", cuda=torch.cuda.is_available())
    lm_model.eval()
    cosine_distance_to_children = []
    all_child_ids = []
    binary_child_text = []
    all_child_embs = []
    all_binary_child_embs = []
    binary_child_embs_mask = np.zeros((len(tree_data), 1))
    for i, row in enumerate(tree_data.to_dict(orient="records")):
        print(i)
        #full_sent, parent_emb, tree_pos, depth = row[1], ast.literal_eval(row[8]), row[6], row[5]
        full_sent, parent_emb, tree_pos, depth, binary_parse = row["full_sent"], vecs[i], row["tree_ind"], row["depth"], row["binary_parse"]
        child_embs, child_ids = get_child_embs(tree_data, full_sent, tree_pos, depth)

        binary_split = get_children_binary_parse(tree_data, full_sent, binary_parse)
        if len(binary_split) == 2:
            left_text, right_text = binary_split
            left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
            right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
            binary_child_embs = torch.stack([left_emb, right_emb]).squeeze(1).unsqueeze(0).cpu().detach().numpy()
            all_binary_child_embs.append(binary_child_embs)
        else:
            binary_child_embs = np.empty((1, 2, 768))
            all_binary_child_embs.append(binary_child_embs)
            binary_child_embs_mask[i] = 1
        #child_embs = get_child_embs_separate(tree_data, full_sent, tree_pos, depth, vecs)
        if len(child_embs) == 0 or len(child_embs) == 1:
            cosine_distance_to_children.append(-1) # leaf node, or unary, mark with -1 to indicate no children
            all_child_ids.append([])
            binary_child_text.append([])
            all_child_embs.append(np.empty((1, 2, 768)))
        else:
            if add_proto_embs:
                proto_emb = ast.literal_eval(row["proto_emb"])
                child_embs = np.vstack([child_embs, np.array(proto_emb)])
                
            child_comp_emb = composition_fn(child_embs)
            cosine_distance_to_children.append(cosine(child_comp_emb, parent_emb))
            all_child_ids.append(child_ids)
            binary_child_text.append(binary_split)
            all_child_embs.append(child_embs)

    all_binary_child_embs = np.concatenate(all_binary_child_embs, axis=0)

    return cosine_distance_to_children, all_child_ids, binary_child_text, all_child_embs, all_binary_child_embs, binary_child_embs_mask

def get_child_embs(tree_data: pd.DataFrame, full_sent: str, tree_pos: tuple, depth: int) -> np.ndarray:
    candidates = tree_data.loc[tree_data["full_sent"] == full_sent]
    if tree_pos == "ROOT":
        children = np.array([ast.literal_eval(emb) for emb in candidates.loc[candidates["depth"] == 1]["emb"]])
        child_ids = [child_id for child_id in candidates.loc[candidates["depth"] == 1]["sent_id"]]
    else:
        right_depth = candidates.loc[candidates["depth"] == depth + 1]
        children = [ast.literal_eval(emb) for emb in right_depth.loc[right_depth["tree_ind"].str.startswith(tree_pos[:-1])]["emb"]]
        child_ids = [child_id for child_id in right_depth.loc[right_depth["tree_ind"].str.startswith(tree_pos[:-1])]["sent_id"]]
        children = np.array(children)
    return children, child_ids

def get_child_embs_from_id(tree_data: pd.DataFrame, child_ids: List) -> np.ndarray:
    children = tree_data.loc[tree_data["sent_id"] in child_ids]
    pdb.set_trace()
    child_embs = np.array([ast.literal_eval(emb) for emb in children["emb"]])
    return child_embs
    
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

def get_children_binary_parse(tree_data: pd.DataFrame, full_sent: str, binary_parse: str) -> tuple:
    binary_tree = Tree.fromstring(binary_parse)
    if len(binary_tree) < 2:
        return []
    left, right = binary_tree[0], binary_tree[1]
    l_text = " ".join(left.leaves())
    r_text = " ".join(right.leaves())
    return l_text, r_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("-i", dest="selection_num", help="process number (split into 10)")
    parser.add_argument("--emb_type", help="embedding type to use", choices=["CLS", "avg", "max"], default="CLS")
    parser.add_argument("--full", help="use all treebank data", action="store_true")
    parser.add_argument("--cuda", help="use gpu", action="store_true")
    parser.add_argument("--use_proto", help="compare to prototype embedding", action="store_true")
    args = parser.parse_args()
    if args.cuda:
        import cupy as np
    else:
        import numpy as np

    i, emb_type, treebank_full = args.selection_num, args.emb_type, args.full
    df = pd.read_csv(f"./tree_data_{i}_{emb_type}_{treebank_full}.csv")

    # Note: sent id is broken for now (not unique...)
    df["sent_id"] = f"P{i}_" + df["sent"].apply(hash).astype(str)

    vecs = np.load(f"./data/embs/{emb_type}_{i}_full_{treebank_full}.npy")

    if args.use_proto:
        df = pd.read_csv("./proto-dist-full.csv")
        vecs = np.load(f"./data/embs/CLS_all_full.npy")

    dist_to_children, child_ids, binary_child_text, all_child_embs, all_binary_child_embs, binary_child_embs_mask = compare_node_to_children(df, vecs, add, add_proto_embs=args.use_proto, vec_type=args.emb_type)
    df["dist_from_children"] = dist_to_children
    df["child_ids"] = child_ids

    # save child embeddings to join to dataset
    # regular children as dict (due to varying lengths), binary children as ndarray
    df["binary_text"] = binary_child_text
    np.savez(f"./data/binary_child_embs_{i}_{emb_type}.npz", child_embs=all_binary_child_embs, mask=binary_child_embs_mask)
    df.to_csv(f"./data/tree_data_{i}_{emb_type}_full_{treebank_full}_proto_{args.use_proto}.csv")