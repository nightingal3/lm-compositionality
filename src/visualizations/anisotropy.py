import pandas as pd
import numpy as np
import pdb
import math
from sklearn.metrics import pairwise_distances
import ast
import matplotlib.pyplot as plt
import seaborn as sns

def sample_rand_vecs(all_vecs: np.ndarray, num_to_sample: int = 50000):
    new_array = [tuple(row) for row in all_vecs]
    all_vecs = np.unique(new_array, axis=0) # don't double count same vecs

    rand_inds = np.random.choice(all_vecs.shape[0], num_to_sample, replace=False)
    rand_sample_rows = all_vecs[rand_inds]
    dist_arr = pairwise_distances(rand_sample_rows, metric="cosine")
    dist_arr_triu = np.triu(dist_arr)
    total_combs = math.comb(num_to_sample, 2)
    cos_dist_sum = np.concatenate(dist_arr_triu).sum()/total_combs

    avg_cos_sim = 1 - cos_dist_sum
    print(avg_cos_sim)
    return avg_cos_sim

def sample_rand_vecs_len_separated(all_vecs: np.ndarray, info_df: pd.DataFrame, num_from_each: int = 5000):
    info_df = info_df.drop_duplicates(subset="sent")
    info_df["word_len"] = info_df["sent"].str.split().apply(len)
    print(info_df["word_len"].value_counts())
    sns.kdeplot(info_df["word_len"])
    plt.savefig("./lengths.png")
    plt.savefig("./lengths.eps")
    assert False
    for i in range(2, 11):
        sel_rows = info_df.loc[info_df["string_len"] == i]
        if len(sel_rows) == 0:
            continue
        all_vecs_sample = np.array([ast.literal_eval(x) for x in sel_rows["emb"]])
        k_sample = min(num_from_each, len(all_vecs_sample))
        avg_cos_sim_i = sample_rand_vecs(all_vecs_sample, k_sample)
        print(f"{i}: {avg_cos_sim_i}")

if __name__ == "__main__":
    info_df = pd.read_csv("/compute/tir-0-17/mengyan3/large_test/roberta_CLS_all_full.csv")
    all_vecs = np.load("/compute/tir-0-17/mengyan3/large_test/parent_embs_roberta_CLS.npy")
    #sample_rand_vecs(all_vecs)
    sample_rand_vecs_len_separated(all_vecs, info_df)