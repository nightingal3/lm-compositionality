import numpy as np
import pandas as pd

def concat_embs(vec_type: str = "cls", model_type: str = "bert") -> None:
    embs_np = [np.load(f"./data/binary_child_embs_{model_type}_{i}_{vec_type}_full_True.npz") for i in range(0, 10)]
    binary_embs = [x["child_embs"] for x in embs_np]
    masks = [x["mask"] for x in embs_np]
    cat_embs = np.concatenate(binary_embs)
    cat_masks = np.concatenate(masks)
    np.save(f"./data/binary_child_embs_{model_type}_{vec_type}.npy", cat_embs)
    np.save(f"./data/binary_child_embs_{model_type}_{vec_type}_mask.npy", cat_masks)

def concat_dfs(vec_type: str = "cls", model_type: str = "bert") -> None:
    df_paths = [f"./data/tree_data_{model_type}_{i}_{vec_type}_full_True_proto_False.csv" for i in range(0, 10)]
    dfs = [pd.read_csv(path) for path in df_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"./data/{model_type}_{vec_type}_all_full.csv", index=False)

if __name__ == "__main__":
    concat_embs("CLS", "roberta")
    concat_dfs("CLS", "roberta")
    assert False
    emb_types = ["CLS", "avg"]
    model_types = ["bert", "roberta", "deberta"]
    for emb_type in emb_types:
        for model_type in model_types:
            print(emb_type, model_type)
            concat_embs(emb_type, model_type)
            concat_dfs(emb_type, model_type)