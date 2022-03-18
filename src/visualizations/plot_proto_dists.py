import pandas as pd
import pdb
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

from plot_compositional_tree_type import filter_rare_trees, record_tree_stats, plot_tree_stats, get_tree_results
from plot_named_entities import plot_cos_dist_hist, filter_rare_entities, plot_named_entity_types, entity_examples

if __name__ == "__main__":
    threshold = 2000
    emb_type = "CLS"
    df = pd.read_csv("./data/tree_data_0_CLS_full_True_proto_True.csv") 
    df_filtered = filter_rare_trees(df, threshold)
    get_tree_results(df_filtered, f"./data/samples/{emb_type}_proto")
    df_filtered.loc[df_filtered["dist_from_children"] >= 0.3].sort_values("dist_from_children", ascending=False)[["sent", "tree_type", "dist_from_children"]].to_csv(f"./high-dist-{emb_type}_proto.csv")
    df.loc[df["dist_from_proto"] >= 0].sort_values("dist_from_proto", ascending=False)[["sent", "tree_type", "dist_from_proto"]].drop_duplicates(subset="sent").to_csv(f"./high-dist-from-proto-{emb_type}_proto.csv")

    tree_stats = record_tree_stats(df_filtered)
    plot_tree_stats(tree_stats, f"tree_dists_{emb_type}_proto.png")

    plt.gcf().clear()
    df = df.loc[df["dist_from_children"] != -1]
    named_entities = df.loc[(df["is_named_entity"] == True) & (df["dist_from_children"] != -1)]
    non_named_entities = df.loc[(df["is_named_entity"] == False) & (df["dist_from_children"] != -1)]
    print(mannwhitneyu(list(named_entities["dist_from_children"]), list(non_named_entities["dist_from_children"])))
    print(mannwhitneyu(list(named_entities["sublength"]), list(non_named_entities["sublength"])))

    plot_cos_dist_hist(df, "ner_test_CLS_proto.png", plot_type="kde")
    plt.gcf().clear()
    named_entities = filter_rare_entities(named_entities, 20)
    plot_named_entity_types(named_entities, "ner_entities_only_CLS_proto.png", plot_type="kde")
    entity_examples(named_entities, "['PERSON']", "person_examples_proto.csv")
    entity_examples(named_entities, "['DATE']", "date_examples_proto.csv")
    entity_examples(named_entities, "['GPE']", "gpe_examples_proto.csv")
    entity_examples(named_entities, "['MONEY']", "money_examples_proto.csv")
    entity_examples(named_entities, "['PERCENT']", "percent_examples_proto.csv")
    entity_examples(named_entities, "['ORG']", "org_examples_proto.csv")
    entity_examples(named_entities, "['CARDINAL']", "cardinal_examples_proto.csv")
    entity_examples(named_entities, "['QUANTITY']", "quantity_examples_proto.csv")
    entity_examples(named_entities, "['LOC']", "loc_examples_proto.csv")
