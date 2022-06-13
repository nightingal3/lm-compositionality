import torch
import os
import pdb

bert_path_CLS = "./data/bert_affine_CLS_results/binary_trees/full"
bert_path_avg = "./data/bert_affine_avg_results/binary_trees/full"
roberta_path_CLS = "./data/roberta_affine_CLS_results/binary_trees/full"
roberta_path_avg = "./data/roberta_affine_avg_results/binary_trees/full"
deberta_path_CLS = "./data/deberta_affine_CLS_results/binary_trees/full"
deberta_path_avg = "./data/deberta_affine_avg_results/binary_trees/full"

if __name__ == "__main__":
    Ws = []
    bs = []
    i = 0
    for filename in os.listdir(deberta_path_avg):
        if not filename.endswith(".pt"):
            continue
        if "original" in filename:
            continue
        full_path = os.path.join(deberta_path_avg, filename)
        weights = torch.load(full_path)
        W, b = weights["W"], weights["b"]
        Ws.append(W)
        bs.append(b)
    mean_W = torch.stack(Ws).squeeze(1).mean(dim=0)
    std_W = torch.stack(Ws).squeeze(1).std(dim=0)
    mean_b = torch.stack(bs).squeeze(1).mean(dim=0)
    print(mean_W)
    print(mean_b)
    pdb.set_trace()