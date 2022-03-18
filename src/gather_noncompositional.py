import pandas as pd
import pdb
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.spatial.distance import cosine

compounds_path = "./data/ijcnlp_compositionality_data/MeanAndDeviations.clean.txt"
BiRD_path = "./data/BiRD/BiRD-annotations-raw.csv"

def get_compounds(compounds_path: str, threshold: float = 2.0) -> pd.DataFrame:
    df = pd.read_csv(compounds_path, sep=" ")
    df[["word2", "word1_mean"]] = df['#word\tWord1_mean'].str.split("\t", expand=True)
    df["word1"] = df.index
    df["word1"] = df["word1"].str.slice(0, -2)
    df["word2"] = df["word2"].str.slice(0, -2)
    df["Cpd"] = df["word1"] + " " + df["word2"]
    df = df.loc[df["Cpd_mean"] <= threshold]
    return df[["word1", "word2", "Cpd", 'Cpd_mean']]

def get_cos_sim_compounds(noncomp_compounds: pd.DataFrame, tokenizer, model) -> None:
    for row in noncomp_compounds.to_dict(orient="records"):
        cpd, w1, w2 = row["Cpd"], row["word1"], row["word2"]
        print(cpd)
        encoded_input = tokenizer(cpd, return_tensors='pt')
        w1_enc = tokenizer(w1, return_tensors="pt")
        w2_enc = tokenizer(w2, return_tensors="pt")
        model_output = model(**encoded_input)[0][0]
        model_output_1 = model(**w1_enc)[0][0]
        model_output_2 = model(**w2_enc)[0][0]
        cls_emb = model_output[0].detach().numpy()
        child_sum = (model_output_1[0] + model_output_1[0]).detach().numpy()
        cos = cosine(cls_emb, child_sum)
        print(cos)

    cpd, w1, w2 = "silver necklace", "silver", "necklace"
    print(cpd)
    encoded_input = tokenizer(cpd, return_tensors='pt')
    w1_enc = tokenizer(w1, return_tensors="pt")
    w2_enc = tokenizer(w2, return_tensors="pt")
    model_output = model(**encoded_input)[0][0]
    model_output_1 = model(**w1_enc)[0][0]
    model_output_2 = model(**w2_enc)[0][0]
    cls_emb = model_output[0].detach().numpy()
    child_sum = (model_output_1[0] + model_output_1[0]).detach().numpy()
    cos = cosine(cls_emb, child_sum)
    print(cos)

if __name__ == "__main__":
    cpd = get_compounds(compounds_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    get_cos_sim_compounds(cpd, tokenizer, model)
