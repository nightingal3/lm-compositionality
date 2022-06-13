import pandas as pd
import matplotlib.pyplot as plt
import pdb
import seaborn as sns

data_path = "./data/probe_perf_long.csv"
model_types = ["bert_cls", "roberta_cls", "deberta_cls", "gpt2_last", 
"bert_avg", "roberta_avg", "deberta_avg", "gpt2_avg"]
sns.set(font_scale=2)
sns.set_theme(style="whitegrid")

if __name__ == "__main__":
    df = pd.read_csv(data_path)
    df["cos_score"] = 1 - df["cos_error"]
    bert_cls = df.loc[df["emb_source"] == "bert_cls"]
    bert_avg = df.loc[df["emb_source"] == "bert_avg"]
    roberta_cls = df.loc[df["emb_source"] == "roberta_cls"]
    roberta_avg = df.loc[df["emb_source"] == "roberta_avg"]
    deberta_cls = df.loc[df["emb_source"] == "deberta_cls"]
    deberta_avg = df.loc[df["emb_source"] == "deberta_avg"]
    gpt2_last = df.loc[df["emb_source"] == "gpt2_last"]
    gpt2_avg = df.loc[df["emb_source"] == "gpt2_avg"]

    fig, axs = plt.subplots(2, 4, figsize=(15, 5))
    
    sns.barplot(ax=axs[0, 0], data=bert_cls, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[0, 0].set_ylim(0.82, 1)
    sns.barplot(ax=axs[1, 0], data=bert_avg, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[1, 0].set_ylim(0.4, 1)
    sns.barplot(ax=axs[0, 1], data=roberta_cls, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[0, 1].set_ylim(0.998, 1)
    sns.barplot(ax=axs[1, 1], data=roberta_avg, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[1, 1].set_ylim(0.92, 1)
    sns.barplot(ax=axs[0, 2], data=deberta_cls, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[0, 2].set_ylim(0.996, 1)
    sns.barplot(ax=axs[1, 2], data=deberta_avg, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[1, 2].set_ylim(0.7, 1)
    sns.barplot(ax=axs[0, 3], data=gpt2_last, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[0, 3].set_ylim(0.94, 1)
    sns.barplot(ax=axs[1, 3], data=gpt2_avg, x="probe", y="cos_score", ci=95, palette="colorblind")
    axs[1, 3].set_ylim(0.97, 1)

    cols = ["BERT", "RoBERTa", "DeBERTa", "gpt2"]
    rows = ["CLS/last token", "AVG"]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=18)

    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, fontsize=18)
    plt.tight_layout()
    plt.savefig("probe_bar.png")
    plt.savefig("probe_bar.eps")