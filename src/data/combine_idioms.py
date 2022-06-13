import pandas as pd
import json
import pdb

if __name__ == "__main__":
    # idioms = []
    # meanings = []
    # with open("./data/idioms/wiki_idiom_collection.json", "r") as input_f:
    #     json_lst = json.load(input_f)
    #     for item in json_lst:
    #         idioms.append(item["idiom_text"])
    #         if len(item["fig_meanings"]) > 0:
    #             meanings.append(item["fig_meanings"][0])
    #         else:
    #             meanings.append("")
    # new_df = pd.DataFrame({"idiom": idioms, "meaning": meanings})
    # old_df = pd.read_csv("./data/idioms/eng.csv")
    # df = pd.concat([new_df, old_df])
    # df.to_csv("./data/idioms/eng_all.csv", index=False)

    
    bigrams_1 = pd.read_csv("./data/ijcnlp_compositionality_data/freq.csv")
    bigrams_2 = pd.read_csv("./data/ramisch2016/compounds-lists/compounds-list-en.tsv", sep="\t")
    all_bigrams = set(list(bigrams_1["compound"])) | set(list(bigrams_2["compound"]))
    pdb.set_trace()