import pandas as pd
from typing import Callable
from scipy.spatial.distance import cosine
import pdb
import ast
from typing import List
import argparse
from nltk import Tree
import random

from src.composition_functions import *
from src.models.get_vecs import get_one_vec, model_init

NULL_VALUES = ["0", "NIL", "*", "*u*", "*ich*", "*exp*", "*rnr*", "*ppa*", "*?*", "-lrb-", "-rrb-"] + [f"*t*-{i}" for i in range(1, 1000)] + [f"*-{i}" for i in range(1, 1000)] + [f"*exp*-{i}" for i in range(1, 1000)] + [f"*ich*-{i}" for i in range(1, 1000)] + [f"*rnr*-{i}" for i in range(1, 1000)] + [f"*ppa*-{i}" for i in range(1, 1000)]
NULL_VALUES = set(NULL_VALUES)

# Gets the cosine distance between a node and some composition of its children.
def compare_node_to_children(tree_data: pd.DataFrame, vecs: np.ndarray, composition_fn: Callable, add_proto_embs: bool = False, vec_type: str = "cls", model_type: str = "encoder", layer: int = 12, use_shuffled: bool = False) -> List:
    lm_model, lm_tokenizer = model_init(model_type, cuda=torch.cuda.is_available())
    model_arch = "gpt" if args.model == "gpt2" else "encoder"

    lm_model.eval()
    cosine_distance_to_children = [] # deprecated
    all_child_ids = [] # deprecated
    binary_child_text = []
    all_child_embs = [] # deprecated
    all_binary_child_embs = []
    binary_child_embs_mask = np.zeros((len(tree_data), 1))
    for i, row in enumerate(tree_data.to_dict(orient="records")):
        print(i)
        #full_sent, parent_emb, tree_pos, depth = row[1], ast.literal_eval(row[8]), row[6], row[5]
        full_sent, parent_emb, tree_pos, depth, binary_parse = row["full_sent"], vecs[i], row["tree_ind"], row["depth"], row["binary_parse"]
        #child_embs, child_ids = get_child_embs(tree_data, full_sent, tree_pos, depth)

        binary_split = get_children_binary_parse(tree_data, full_sent, binary_parse, use_shuffled=use_shuffled)

        if len(binary_split) == 2:
            left_text, right_text = binary_split
            left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, model_type=model_arch, cuda=torch.cuda.is_available(), layer=layer)
            right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, model_type=model_arch,cuda=torch.cuda.is_available(), layer=layer)
            binary_child_embs = torch.stack([left_emb, right_emb]).squeeze(1).unsqueeze(0).cpu().detach().numpy()

            all_binary_child_embs.append(binary_child_embs)
            binary_child_text.append(binary_split)
        else:
            binary_child_embs = np.empty((1, 2, 768))
            all_binary_child_embs.append(binary_child_embs)
            binary_child_embs_mask[i] = 1
            binary_child_text.append([])

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

def get_children_binary_parse(tree_data: pd.DataFrame, full_sent: str, binary_parse: str, use_shuffled: bool = False) -> tuple:
    binary_tree = Tree.fromstring(binary_parse)
    if len(binary_tree) < 2:
        return []
    left, right = binary_tree[0], binary_tree[1]
    l_text = " ".join([leaf for leaf in left.leaves() if leaf.lower() not in NULL_VALUES])
    r_text = " ".join([leaf for leaf in right.leaves() if leaf.lower() not in NULL_VALUES])
    if len(l_text) == 0 or len(r_text) == 0: # only null value in left or right branch
        return []
    
    if use_shuffled:
        full_text = (l_text + " " + r_text).split()
        random.shuffle(full_text)
        l_text = " ".join(full_text[:len(l_text.split())])
        r_text = " ".join(full_text[len(l_text.split()):])

    return l_text, r_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("--model", help="model type to examine", choices=["bert", "roberta", "deberta", "gpt2"], default="bert")
    parser.add_argument("-i", dest="selection_num", help="process number (split into 10)")
    parser.add_argument("--emb_type", help="embedding type to use", choices=["CLS", "avg", "max"], default="CLS")
    parser.add_argument("--full", help="use all treebank data", action="store_true")
    parser.add_argument("--use_proto", help="compare to prototype embedding", action="store_true")
    parser.add_argument("--layer", help="layer to compute", choices=[i for i in range(1, 13)], type=int)
    parser.add_argument("--control_task", help="shuffle the words between left and right child", action="store_true")
    args = parser.parse_args()

    i, emb_type, treebank_full = args.selection_num, args.emb_type, args.full
    df = pd.read_csv(f"./data/raw_tree_data/tree_data_{args.model}_{i}_{emb_type}_{treebank_full}_layer_{args.layer}.csv")

    df["sent_id"] = f"P{i}_" + df["sent"].apply(hash).astype(str)

    vecs = np.load(f"./data/embs/{args.model}_{emb_type}_{i}_full_{treebank_full}_layer_{args.layer}.npy")

    if args.use_proto:
        df = pd.read_csv("./proto-dist-full.csv")
        vecs = np.load(f"./data/embs/CLS_all_full.npy")

    dist_to_children, child_ids, binary_child_text, all_child_embs, all_binary_child_embs, binary_child_embs_mask = compare_node_to_children(df, vecs, add, add_proto_embs=args.use_proto, vec_type=args.emb_type, model_type=args.model, layer=args.layer, use_shuffled=args.control_task)
    #df["dist_from_children"] = dist_to_children
    #df["child_ids"] = child_ids

    # save child embeddings to join to dataset
    # regular children as dict (due to varying lengths), binary children as ndarray
    df["binary_text"] = binary_child_text

    if args.control_task:
        emb_filename = f"./data/binary_child_embs_{args.model}_{i}_{emb_type}_full_{args.full}_layer_{args.layer}_SHUFFLE.npz"
        df_filename = f"./data/tree_data_processed/tree_data_{args.model}_{i}_{emb_type}_full_{treebank_full}_proto_{args.use_proto}_layer_{args.layer}_SHUFFLE.csv"
    else:
        emb_filename = f"./data/binary_child_embs_{args.model}_{i}_{emb_type}_full_{args.full}_layer_{args.layer}.npz"
        df_filename = f"./data/tree_data_processed/tree_data_{args.model}_{i}_{emb_type}_full_{treebank_full}_proto_{args.use_proto}_layer_{args.layer}.csv"

    np.savez(emb_filename, child_embs=all_binary_child_embs, mask=binary_child_embs_mask)
    df.to_csv(df_filename, index=False)