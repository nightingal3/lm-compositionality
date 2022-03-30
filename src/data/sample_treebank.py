import pandas as pd
import pdb

data_file = "./data/CLS_all_full.csv"

if __name__ == "__main__":
    df = pd.read_csv(data_file, engine="python", columns=["dist_from_children", "sublength", "sent", "binary_text", "is_idiom"])
    non_leaves = df.loc[((df["dist_from_children"] != -1) & (df["dist_from_children"].notnull())) & ((df["sublength"] <= 5) & (df["sublength"] > 2))]
    sample = non_leaves.sample(n=5000, random_state=42)
    idioms = non_leaves.loc[non_leaves["is_idiom"] == True]
    sample = sample.append(idioms)
    sample_df = sample[["sent", "binary_text", "is_idiom", "sublength"]]

    pdb.set_trace()
    sample_df.to_csv("./data/human_experiments/sample_phrases_treebank.csv", index=False)