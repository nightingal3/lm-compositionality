import seaborn as sns
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf

from plot_compositional_tree_type import merge_dataframes

def plot_cos_dist_hist(df: pd.DataFrame, out_filename: str, dim: int = 1) -> None:
    if dim == 1:
        sns.histplot(data=df, x="dist_from_children", hue="is_named_entity", kde=True)
    else:
        sns.histplot(data=df, x="sublength", y="dist_from_children", hue="is_named_entity", kde=True)

    plt.savefig(out_filename)

def plot_named_entity_types(named_entities: pd.DataFrame, out_filename: str, dim: int = 1) -> None:
    if dim == 1:
        sns.histplot(data=named_entities, x="dist_from_children", hue="named_ent_types", kde=True, alpha=0.5)
    else:
        sns.histplot(data=named_entities, x="sublength", y="dist_from_children", hue="named_ent_types", kde=True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_filename)

def filter_rare_entities(named_entities: pd.DataFrame, threshold: int) -> pd.DataFrame:
    return named_entities.groupby("named_ent_types").filter(lambda x: len(x) >= threshold)

def glm_model(data: pd.DataFrame) -> None:
    md = smf.mixedlm("dist_from_children ~ sublength + is_named_entity", data, groups=data["sublength"], re_formula="~is_named_entity")
    mdf = md.fit(method=["lbfgs"])
    print(mdf.summary())

def entity_examples(data: pd.DataFrame, cat: str, out_filename: str) -> None:
    data = data.loc[data["named_ent_types"] == cat]
    data = data.drop_duplicates(subset=["sent"])
    data = data.sort_values(by="dist_from_children")[["sent", "dist_from_children"]]
    data.to_csv(out_filename, index=False)

if __name__ == "__main__":
    emb_type = "CLS"
    df_paths = [f"./data/tree_data_{i}_{emb_type}.csv" for i in range(0, 10)]
    dfs = [pd.read_csv(path) for path in df_paths]
    df = merge_dataframes(dfs)
    df = df.loc[df["dist_from_children"] != -1]
    named_entities = df.loc[(df["is_named_entity"] == True) & (df["dist_from_children"] != -1)]
    non_named_entities = df.loc[(df["is_named_entity"] == False) & (df["dist_from_children"] != -1)]
    print(mannwhitneyu(list(named_entities["dist_from_children"]), list(non_named_entities["dist_from_children"])))
    print(mannwhitneyu(list(named_entities["sublength"]), list(non_named_entities["sublength"])))

    plot_cos_dist_hist(df, "ner_test_CLS.png")
    plt.gcf().clear()
    named_entities = filter_rare_entities(named_entities, 20)
    plot_named_entity_types(named_entities, "ner_entities_only_CLS.png")
    glm_model(df)
    entity_examples(named_entities, "['PERSON']", "person_examples.csv")
    entity_examples(named_entities, "['DATE']", "date_examples.csv")
    entity_examples(named_entities, "['GPE']", "gpe_examples.csv")
    entity_examples(named_entities, "['MONEY']", "money_examples.csv")
    entity_examples(named_entities, "['PERCENT']", "percent_examples.csv")
    entity_examples(named_entities, "['ORG']", "org_examples.csv")
    entity_examples(named_entities, "['CARDINAL']", "cardinal_examples.csv")
    entity_examples(named_entities, "['QUANTITY']", "quantity_examples.csv")



