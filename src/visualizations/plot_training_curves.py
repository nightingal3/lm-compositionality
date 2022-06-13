import numpy as np
from sklearn.metrics import auc
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb

model = "deberta"
emb_type = "CLS"

training_curves_path_lin = f"./data/{model}_linear_{emb_type}_results/binary_trees/full/"
training_curves_path_aff = f"./data/{model}_affine_{emb_type}_results/binary_trees/full/"
training_curves_path_mlp = f"./data/{model}_mlp_{emb_type}_results/binary_trees/full/"

nonpar_probes_results = {
    "bert": {
        "add": {
            "CLS": 0.0819,
            "avg": 0.1796
        },
        "w1": {
            "CLS": 0.1618,
            "avg": 0.4617   
        },
        "w2": {
            "CLS": 0.08793,
            "avg": 0.1189
        }
    },
    "roberta": {
        "add": {
            "CLS": 0.0006519,
            "avg": 0.02482
        },
        "w1": {
            "CLS": 0.001499,
            "avg": 0.07262
        }, 
        "w2": {
            "CLS": 0.0006320,
            "avg": 0.01505
        }
    },
    "deberta": {
        "add": {
            "CLS": 0.0009222,
            "avg": 0.07251
        },
        "w1": {
            "CLS": 0.002376,
            "avg": 0.2632
        },
        "w2": {
            "CLS": 0.0008935,
            "avg": 0.05619
        }
    },
    "gpt2": {
        "add": {
            "CLS": 0.03838,
            "avg": 0.003971
        }, 
        "w1": {
            "CLS": 0.04054,
            "avg": 0.020892
        },
        "w2": {
            "CLS": 0.04025,
            "avg": 0.003524
        }
    }
}
def plot_training_curves(out_filename: str = "training_curves_sample.png") -> None:
    # Milestones: 0.005%, 0.01%, 0.1%, 0.5%, 1%, 10%, 100% increments 

    x = np.array([0.005, 0.01, 0.1, 0.5, 1, 10, 100])
    
    data_increments = []
    probe_types = []
    score = []
    lin_scores = np.zeros((9, 7))
    mlp_scores = np.zeros((9, 7))
    aff_scores = np.zeros((9, 7))
    add_scores = np.zeros((9, 7))
    w1_scores = np.zeros((9, 7))
    w2_scores = np.zeros((9, 7))

    for i in range(9):
        path_lin = os.path.join(training_curves_path_lin, f"model_{i}_losses.p")
        path_mlp = os.path.join(training_curves_path_mlp, f"model_{i}_losses.p")
        path_aff = os.path.join(training_curves_path_aff, f"model_{i}_losses.p")

        y_lin_i = pickle.load(open(path_lin, "rb"))
        y_mlp_i = pickle.load(open(path_mlp, "rb"))
        y_aff_i = pickle.load(open(path_aff, "rb"))
        y_add_i = [nonpar_probes_results[model]["add"][emb_type]] * 7
        y_w1_i = [nonpar_probes_results[model]["w1"][emb_type]] * 7
        y_w2_i = [nonpar_probes_results[model]["w2"][emb_type]] * 7

        y_lin_i = [1 - item for item in y_lin_i]
        y_mlp_i = [1 - item for item in y_mlp_i]
        y_aff_i = [1 - item for item in y_aff_i]
        y_add_i = [1 - item for item in y_add_i]
        y_w1_i = [1 - item for item in y_w1_i]
        y_w2_i = [1 - item for item in y_w2_i]

        lin_scores[i, :] = y_lin_i
        mlp_scores[i, :] = y_mlp_i
        aff_scores[i, :] = y_aff_i
        add_scores[i, :] = y_add_i
        w1_scores[i, :] = y_w1_i
        w2_scores[i, :] = y_w2_i

        for j in range(len(x)):
            data_increments.extend([j] * 6)
            probe_types.append("add")
            score.append(y_add_i[j])
            probe_types.append("lin")
            score.append(y_lin_i[j])
            probe_types.append("aff")
            score.append(y_aff_i[j])
            probe_types.append("mlp")
            score.append(y_mlp_i[j])
            probe_types.append("w1")
            score.append(y_w1_i[j])
            probe_types.append("w2")
            score.append(y_w2_i[j])
    data_df = pd.DataFrame({"percent_dataset": data_increments, "probe_types": probe_types, "cos_score": score})
    sns.lineplot(data=data_df, x="percent_dataset", y="cos_score", hue="probe_types", marker="o")
    ax = plt.gca()
    ax.xaxis.set_ticks(range(len(x)))
    ax.xaxis.set_ticklabels([0.005, 0.01, 0.1, 0.5, 1, 10, 100])
    plt.savefig(f"{out_filename}.eps")
    plt.savefig(f"{out_filename}.png")

    y_lin = lin_scores.mean(axis=0)
    y_mlp = mlp_scores.mean(axis=0)
    y_aff = aff_scores.mean(axis=0)
    y_add = add_scores.mean(axis=0)
    y_w1 = w1_scores.mean(axis=0)
    y_w2 = w2_scores.mean(axis=0)

    print("lin", auc(x, y_lin))
    print("mlp", auc(x, y_mlp))
    print("aff", auc(x, y_aff))
    print("add", auc(x, y_add))
    print("w1", auc(x, y_w1))
    print("w2", auc(x, y_w2))

if __name__ == "__main__":
    plot_training_curves(f"training_curves_{model}_{emb_type}")
