import pandas as pd
import numpy as np
import pdb
from typing import List
import ast
import json

def generate_roles_left_right(strings: List) -> List:
    return [[i for i in range(len(s))] for s in strings]
x
def generate_roles_right_left(strings: List) -> List:
    return [[i for i in reversed(range(len(s)))] for s in strings]

def generate_bow_roles(strings: List) -> List:
    return [[0] * len(s) for s in strings]

def generate_tree_roles(strings: List, tree_positions: List) -> List:
    raise NotImplementedError

def write_jsonl(df: pd.DataFrame, out_filename: str) -> None:
    records = []
    for i, row in enumerate(df.to_dict(orient="records")):
        records.append({"linex_index": i, "sentence": row["sent"], "parse": row["parse"], "binaryParse": row["binary_parse"], "pooled_output": ast.literal_eval(row["emb"])})
    with open(out_filename, "w") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    df = pd.read_csv("./tree_data_0_CLS_False.csv")
    df = df.dropna(subset=["sent"]) # drop null trees
    strings = list(df["sent"])
    vecs = [ast.literal_eval(emb) for emb in list(df["emb"])]

    l_r_roles = generate_roles_left_right(strings)
    r_l_roles = generate_roles_right_left(strings)
    bow_roles = generate_bow_roles(strings)
    write_jsonl(df, "test.jsonl")
    
