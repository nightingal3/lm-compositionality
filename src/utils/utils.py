import numpy as np

def concat_embs(vec_type: str = "cls") -> None:
    embs_np = [np.load(f"./data/binary_child_embs_{i}.npz") for i in range(0, 10)]
    binary_embs = [x["child_embs"] for x in embs_np]
    masks = [x["mask"] for x in embs_np]
    cat_embs = np.concatenate(binary_embs)
    cat_masks = np.concatenate(masks)
    np.save(f"./data/binary_child_embs_{vec_type}.npy", cat_embs)
    np.save("./data/binary_child_embs_cls_mask.npy", cat_masks)

if __name__ == "__main__":
    concat_embs()