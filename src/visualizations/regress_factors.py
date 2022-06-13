import pandas as pd
import pdb
from scipy.stats import spearmanr

if __name__ == "__main__":
    model = "bert"
    emb_type = "CLS"
    path = f"./data/{model}_affine_{emb_type}_results/binary_trees/full/deviations_all.csv"

    df = pd.read_csv(path)
    df["total_len"] = df["sent"].apply(lambda x: len(x.split()))
    df["left_len"] = df["binary_text"].apply(lambda x: len(x.split(",")[0][1:].split()))
    df["right_len"] = df["binary_text"].apply(lambda x: len(x.split(",")[1][:-1].split()))
    print(spearmanr(df["total_len"], df["deviation"]))
    print(spearmanr(df["left_len"], df["deviation"]))
    print(spearmanr(df["right_len"], df["deviation"]))