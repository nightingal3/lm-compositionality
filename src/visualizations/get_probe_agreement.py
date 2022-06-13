import pandas as pd
import pdb
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = "bert"
    emb_type = "CLS"

    add_paths = [f"./data/{model}_add_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
    w1_paths = [f"./data/{model}_w1_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
    w2_paths = [f"./data/{model}_w2_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
    aff_paths = [f"./data/{model}_affine_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
    lin_paths = [f"./data/{model}_linear_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]
    #mlp_paths = [f"./data/{model}_mlp_{emb_type}_results/binary_trees/full/deviations_{i}.csv" for i in range(0, 10)]

    df_add = pd.concat([pd.read_csv(x) for x in add_paths]).reset_index(drop=True).sort_values(by="sent")
    df_add.to_csv(f"./data/{model}_add_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)
    df_w1 = pd.concat([pd.read_csv(x) for x in w1_paths]).reset_index(drop=True).sort_values(by="sent")
    df_w1.to_csv(f"./data/{model}_w1_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)
    df_w2 = pd.concat([pd.read_csv(x) for x in w2_paths]).reset_index(drop=True).sort_values(by="sent")
    df_w2.to_csv(f"./data/{model}_w2_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)
    df_aff = pd.concat([pd.read_csv(x) for x in aff_paths]).reset_index(drop=True).sort_values(by="sent")
    df_aff.to_csv(f"./data/{model}_affine_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)
    assert False
    df_lin = pd.concat([pd.read_csv(x) for x in lin_paths]).reset_index(drop=True).sort_values(by="sent")
    df_lin.to_csv(f"./data/{model}_linear_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)
    df_mlp = pd.concat([pd.read_csv(x) for x in mlp_paths]).reset_index(drop=True).sort_values(by="sent")
    df_mlp.to_csv(f"./data/{model}_mlp_{emb_type}_results/binary_trees/full/deviations_all.csv", index=False)

    dfs = [df_add, df_w1, df_w2, df_aff, df_lin, df_mlp]

    #joined = reduce(lambda l, r: pd.merge(l, r, on="sent", how="inner"), dfs)
    rank_add = df_add["deviation"].rank().tolist()
    rank_w1 = df_w1["deviation"].rank().tolist()
    rank_w2 = df_w2["deviation"].rank().tolist()
    rank_aff = df_aff["deviation"].rank().tolist()
    rank_lin = df_lin["deviation"].rank().tolist()
    rank_mlp = df_mlp["deviation"].rank().tolist()
    rank_df = pd.DataFrame({"add": rank_add, "w1": rank_w1, "w2": rank_w2, "affine": rank_aff, "linear": rank_lin, "mlp": rank_mlp})
    corr = rank_df.corr(method="kendall")
    mask = np.triu(np.ones_like(rank_df.corr()))
    sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='icefire')
    plt.savefig(f"heatmap_{model}_{emb_type}.png")
    plt.savefig(f"heatmap_{model}_{emb_type}.eps")

