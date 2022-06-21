import ast
import pandas as pd
import numpy as np
import torch
from typing import Callable
from datasets import load_dataset, concatenate_datasets, Dataset
import argparse
import pdb
from tqdm import tqdm
from pathlib import Path
import time
from tempfile import mkdtemp
import os
from sklearn.model_selection import KFold
import pickle
import random

from src import composition_functions
from src.calc_compositionality_scores import get_child_embs, get_child_embs_from_id
from src.models.get_vecs import get_one_vec, model_init

probe_types = {
    "add": "nonpar",
    "mult": "nonpar",
    "w1": "nonpar",
    "w2": "nonpar",
    "linear": "par",
    "affine": "par",
    "mlp": "par",
    "tpdn": "par"
}
MAX_EPOCHS = 20 
BATCH_SIZE = 512

SCRATCH_DIR = "/compute/tir-0-19/mengyan3"

# for composition prediction, input size is number of children and output size is embedding size.
# for decomposition prediction, input size is embedding size and output size is number of children * embedding size
class AffineRegression(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, cuda: bool = False):
        super(AffineRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = "cuda" if cuda else "cpu"
        self.W = torch.nn.Parameter(torch.randn((1, input_size), requires_grad=True, device=self.device))
        self.b = torch.nn.Parameter(torch.randn((1, output_size), requires_grad=True, device=self.device))

    def forward(self, x): 
        pred = torch.matmul(self.W, x) + self.b
        return pred.squeeze(1)

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, cuda: bool = False):
        super(LinearRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = "cuda" if cuda else "cpu"
        self.W = torch.nn.Parameter(torch.randn((1, input_size), requires_grad=True, device=self.device))

    def forward(self, x): 
        pred = torch.matmul(self.W, x) # (1 x 2) x (b x 2 x 768)
        return pred.squeeze(1) # (b x 768)

class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, cuda: bool = False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = "cuda" if cuda else "cpu"
        self.hidden1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.output = torch.nn.Linear(self.output_size, 1)
        # reasonable number of layers?

    def forward(self, x):
        # x: (b, 2, 768)
        x_t = torch.transpose(x, 1, 2)
        hidden = self.hidden1(x_t)
        hidden_relu = self.relu(hidden)
        hidden_2 = self.hidden2(hidden_relu)
        hidden_2_relu = self.relu(hidden_2)
        out = self.output(hidden_2_relu)
        return out.squeeze(2)

class EmbeddingDataset(Dataset):
    def __init__(self, csv_path, child_embeddings_path, parent_embeddings_path):
        self.csv_path = csv_path
        self.child_embeddings_path = child_embeddings_path
        self.parent_embeddings_path = parent_embeddings_path
        self.raw_data_df = pd.read_csv(csv_path)
        self.raw_data_df["original_order"] = self.raw_data_df.index
        self.data_df = self.raw_data_df.loc[self.raw_data_df["binary_text"].str.len() > 2] # () has length 2
        self.data_df = self.data_df.drop_duplicates(subset=["sent"])
        self.sents_with_children = set(self.data_df["sent"])
        self.binary_child_vec = np.load(child_embeddings_path, mmap_mode="r")
        self.parent_vec = np.load(parent_embeddings_path, mmap_mode="r")
        #self.parent_embs = self.data_df["emb"].apply(lambda x: ast.literal_eval(x))

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        sent = self.data_df["sent"].iloc[idx]
        tree_type = self.data_df["tree_type"].iloc[idx]
        binary_tree_type = self.data_df["binary_tree_type"].iloc[idx] if "binary_tree_type" in self.data_df.columns else None
        binary_text = self.data_df["binary_text"].iloc[idx]
        original_index = self.data_df["original_order"].iloc[idx]
        #parent_emb = torch.tensor(ast.literal_eval(self.data_df["emb"].iloc[idx]), dtype=torch.float32)
        parent_emb = torch.tensor(self.parent_vec[original_index], dtype=torch.float32)
        child_embs = torch.tensor(self.binary_child_vec[original_index], dtype=torch.float32)

        return {
            "sent": sent,
            "tree_type": tree_type,
            "binary_tree_type": binary_tree_type,
            "binary_text": binary_text,
            "emb": parent_emb,
            "child_embs": child_embs
        }        

# From original TPDN code!
# https://github.com/tommccoy1/tpdn/blob/a4ea54030056a49e5fd00a700eb71790157bc697/binding_operations.py#L13
class SumFlattenedOuterProduct(torch.nn.Module):
    def __init__(self):
        super(SumFlattenedOuterProduct, self).__init__()
           
    def forward(self, input1, input2): # (batch length x vec size x role size)
        input1 = input1.unsqueeze(-1)
        input2 = input2.unsqueeze(1)
        outer_product = torch.bmm(input1, input2)
        sum_outer_product = torch.sum(outer_product, dim=0)
        flattened_outer_product = torch.flatten(sum_outer_product).unsqueeze(0)
        return flattened_outer_product

class TPDN(torch.nn.Module): #TODO: make it possible to initialize the roles with pretrained embeddings
    def __init__(self, input_size: int, output_size: int, filler_size: int, role_size: int, role_type: str = "index", cuda: bool = False):
        super(TPDN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.filler_size = filler_size
        self.role_size = role_size
        self.device = "cuda" if cuda else "cpu"
        if role_type == "pos": # all treebank tags
            self.input_size = 48

        #self.filler_emb = torch.nn.Embedding(self.input_size, self.filler_size)
        self.role_emb = torch.nn.Embedding(self.input_size, self.role_size, device=self.device)
        self.sum_layer = SumFlattenedOuterProduct()
        self.out_linear = torch.nn.Linear(self.filler_size * self.role_size, self.output_size, device=self.device)
            
    def forward(self, x, roles):
        role_emb = self.role_emb(roles)
        summed = self.sum_layer(x, role_emb)
        out = self.out_linear(summed).squeeze(0)
        return out

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def fit_composition(probe_type: str = "add", vec_type: str = "CLS", loss_type: str = "cosine", rand_seed: int = 42, full: bool = False, binary_trees: bool = False, control_task: bool = False, use_trained: bool = False, lm_type: str = "bert") -> None:
    seed_everything(rand_seed)
    seed_seq = np.random.randint(0, 1000, 10)
    print(seed_seq)

    log_path =  f"./data/{lm_type}_{probe_type}_{vec_type}_results/binary_trees/" if binary_trees else f"./data/{probe_type}_{vec_type}_results/"
    vec_type_lower = vec_type.lower()

    if control_task: 
        log_path += "control_setting"

    if full:
        log_path += "full"
        #data_df = pd.read_csv(f"./data/{lm_type}_{vec_type}_all_full.csv")
        #data_df["original_order"] = data_df.index
        #dataset_full = Dataset.from_pandas(data_df)
        dataset_full = EmbeddingDataset(f"{SCRATCH_DIR}/large_test/{lm_type}_{vec_type}_all_full.csv", f"{SCRATCH_DIR}/large_test/binary_child_embs_{lm_type}_{vec_type_lower}.npy", f"{SCRATCH_DIR}/large_test/parent_embs_{lm_type}_{vec_type}.npy")
        kfold = KFold(n_splits=10, shuffle=False)
        dataset_full_inds = kfold.split(dataset_full)

    else:
        data_df = pd.read_csv(f"./data/small_test/{vec_type}_sample.csv")
        dataset_full = EmbeddingDataset(f"./data/small_test/{vec_type}_sample.csv", f"./data/small_test/{vec_type}_bin_data_sample.npy", f"./data/small_test/{vec_type}")
        # loading datasets using load_dataset and split doesn't work for some reason....
        kfold = KFold(n_splits=10, shuffle=False)
        dataset_full_inds = kfold.split(dataset_full)

    if loss_type == "cosine":
        loss_fn = lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=1)
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()

    overall_loss = {}
    overall_loss_cats = {}
    total_test_loss_cats = {}
    partition_len = {}
    freq_cats_overall = {}


    for i, (train_indices, test_indices) in enumerate(dataset_full_inds):
        print(f"=== PARTITION {i} ===")
        if i != 7:
            continue
        seed_everything(seed_seq[i])
        if probe_types[probe_type] == "par":
            random.shuffle(test_indices)
            val_indices, true_test_indices = test_indices[: len(test_indices) // 2], test_indices[len(test_indices) // 2:]
            test_data = torch.utils.data.Subset(dataset_full, true_test_indices.tolist())
            val_data = torch.utils.data.Subset(dataset_full, val_indices.tolist())
            train_data = torch.utils.data.Subset(dataset_full, train_indices.tolist())
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        else:
            train_data = []
            random.shuffle(test_indices)
            val_indices, true_test_indices = test_indices[: len(test_indices) // 2], test_indices[len(test_indices) // 2:]
            test_data = torch.utils.data.Subset(dataset_full, true_test_indices.tolist())
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        print("~TRAIN~")
        if probe_types[probe_type] == "par":
            model = train(i, probe_type, train_loader, val_loader, loss_fn, log_path, binary_trees, record_training_curve=True, train_dataset=train_data)
        else:
            model = None

        print("~TEST~")
        num_samples = len(test_loader.dataset)
        test_loss, test_loss_cats, freq_cats, deviations_from_expected = test(model, probe_type, test_loader, loss_fn, log_path, binary_trees, control_task=control_task, vec_type=vec_type, record_deviation=True, full_dataset=full)
        deviation_df = pd.DataFrame(deviations_from_expected).sort_values(by="deviation")
        Path(log_path).mkdir(parents=True, exist_ok=True)
        deviation_df.to_csv(f"{log_path}/deviations_{i}.csv", index=False)
        
        avg_test_loss = test_loss / num_samples
        avg_test_loss_cats = {cat: loss / freq_cats[cat] for cat, loss in test_loss_cats.items()}
        overall_loss[i] = avg_test_loss
        partition_len[i] = num_samples

        for cat in avg_test_loss_cats: # feel like there should be a builtin to efficiently do this?
            if cat not in overall_loss_cats:
                overall_loss_cats[cat] = avg_test_loss_cats[cat]
            else:
                overall_loss_cats[cat] += avg_test_loss_cats[cat]
        for cat in freq_cats: # feel like there should be a builtin to efficiently do this?
            if cat not in freq_cats_overall:
                freq_cats_overall[cat] = freq_cats[cat]
            else:
                freq_cats_overall[cat] += freq_cats[cat]
        for cat in test_loss_cats:
            if cat not in total_test_loss_cats:
                total_test_loss_cats[cat] = test_loss_cats[cat]
            else:
                total_test_loss_cats[cat] += test_loss_cats[cat]

        Path(log_path).mkdir(parents=True, exist_ok=True)
        with open(f"{log_path}/results_{probe_type}_{i}.txt", "w") as out_f:
            out_f.write(f"Average test loss: {avg_test_loss}\n")
            out_f.write("===CATEGORIES===\n")
            for cat in sorted(avg_test_loss_cats, key=lambda x: freq_cats[x], reverse=True):
                out_f.write(f"{cat}: {avg_test_loss_cats[cat]}\n")
            
    avg_test_loss_overall = sum([overall_loss[i] * partition_len[i] for i in range(10)])/sum([partition_len[i] for i in range(10)])
    
    with open(f"{log_path}/results_{probe_type}_ALL.txt", "w") as out_f:
        out_f.write(f"Average test loss: {avg_test_loss_overall}\n")
        out_f.write("===CATEGORIES===\n")
        for cat in sorted(overall_loss_cats, key=lambda x: freq_cats_overall[cat], reverse=True):
            out_f.write(f"{cat}: {total_test_loss_cats[cat] / freq_cats_overall[cat]},{freq_cats_overall[cat]}\n")


def train(partition_num: int, probe_type: str, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, loss_fn: Callable, log_path: str, binary_trees: bool, record_training_curve: bool = False, patience: int = 2, train_dataset = None):
    all_children = []
    num_samples = len(train_dataset)
    num_samples_dev = len(test_loader.dataset) 
    if probe_type == "linear":
            if binary_trees:
                model = LinearRegression(input_size=2, output_size=768, cuda=torch.cuda.is_available())
            else:
                model = LinearRegression(input_size=5, output_size=768, cuda=torch.cuda.is_available())
    elif probe_type == "affine":
        if binary_trees:
            model = AffineRegression(input_size=2, output_size=768, cuda=torch.cuda.is_available())
        else:
            model = AffineRegression(input_size=5, output_size=768, cuda=torch.cuda.is_available())
    elif probe_type == "mlp":
        if binary_trees:
            model = MLP(input_size=2, output_size=768, hidden_size=300, cuda=torch.cuda.is_available())
        else:
            model = MLP(input_size=5, output_size=768, hidden_size=300, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            model.to("cuda")
    elif probe_type == "tpdn":
        if binary_trees:
            model = TPDN(input_size=2, output_size=768, hidden_size=300, role_size=10, cuda=torch.cuda.is_available())
        else:
            model = TPDN(input_size=5, output_size=768, filler_size=768, role_size=10, cuda=torch.cuda.is_available())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * BATCH_SIZE)

    # Save the original weights so we can compare data size for the same weight initialization
    Path(log_path).mkdir(parents=True, exist_ok=True)
    model_name_orig = f"model_{partition_num}_bintree_original" if binary_trees else f"model_{partition_num}_original"  
    torch.save(model.state_dict(), f"{log_path}/{model_name_orig}.pt")

    # Milestones: 0.005%, 0.01%, 0.1%, 0.5%, 1%, 10%, 100% increments 
    if record_training_curve:
        training_milestones = [int(0.00005 * num_samples), int(0.0001 * num_samples), int(0.001 * num_samples), int(0.005 * num_samples), int(0.01 * num_samples),  int(0.1 * num_samples), num_samples]
    else:
        training_milestones = [num_samples]

    milestone_losses = []

    for i, dataset_size in enumerate(training_milestones):
        if i < 4:
            continue
        if i == 4:
            pdb.set_trace()
        print("DATASET SIZE: ", dataset_size)
        epochs_no_improvement = 0
        best_loss = float("inf")
        torch.cuda.empty_cache()
        if i > 0: # reset to original params
            model.load_state_dict(torch.load(f"{log_path}/{model_name_orig}.pt"))

        if dataset_size != num_samples:
            random_indices = random.sample(list(range(num_samples)), dataset_size)
            train_subset = torch.utils.data.Subset(train_dataset, random_indices)
            subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        train_loader_subset = train_loader if dataset_size == num_samples else subset_loader

        for epoch in range(MAX_EPOCHS):
            if epochs_no_improvement == patience:
                print("no improvements seen. Stopping training...")
                break
            print(f"EPOCH {epoch}")
            epoch_loss_total = 0
            for sample in tqdm(train_loader_subset):
                child_embs = sample["child_embs"]
                actual_emb = sample["emb"]
                if torch.cuda.is_available():
                    child_embs = child_embs.to("cuda")
                    actual_emb = actual_emb.to("cuda")

                pred = model(child_embs)

                optimizer.zero_grad()
                if probe_type == "tpdn":
                    roles = torch.tensor(range(len(child_embs)), device="cuda", dtype=int)
                    pred = model(child_embs, roles)
                else:
                    pred = model(child_embs)
                loss = loss_fn(actual_emb, pred)
                mean_loss = torch.mean(loss)
                total_loss = torch.sum(loss)
                epoch_loss_total += total_loss.item()
                mean_loss.backward()
                optimizer.step()

            dev_loss, dev_loss_cats, freq_cats, _ = test(model, probe_type, test_loader, loss_fn, log_path, binary_trees)
            model.train()
    
            avg_dev_loss = dev_loss/num_samples_dev
            #print(model.W)
            print(avg_dev_loss)

            if avg_dev_loss < best_loss:
                print("improvement :)")
                epochs_no_improvement = 0
                best_loss = avg_dev_loss

                if probe_types[probe_type] == "par":
                    Path(log_path).mkdir(parents=True, exist_ok=True)
                    model_name = f"model_{partition_num}_bintree" if binary_trees else f"model_{partition_num}"  
                    torch.save(model.state_dict(), f"{log_path}/{model_name}.pt")
            else:
                print("no improvement :(")
                epochs_no_improvement += 1

        model.load_state_dict(torch.load(f"{log_path}/{model_name}.pt"))
        dev_loss, dev_loss_cats, freq_cats, _ = test(model, probe_type, test_loader, loss_fn, log_path, binary_trees)
        model.train()

        avg_dev_loss = dev_loss/num_samples_dev
        milestone_losses.append(avg_dev_loss)

    if record_training_curve:
        losses_name = f"{log_path}/model_{partition_num}_losses.p"
        with open(losses_name, "wb") as loss_file:
            pickle.dump(milestone_losses, loss_file)

    return model

def test(model, probe_type: str, test_loader: torch.utils.data.DataLoader, loss_fn: Callable, log_path: str, binary_trees: bool, control_task: bool = False, lm_model = None, lm_tokenizer = None, vec_type: str = "cls", record_deviation=False, full_dataset: bool = True):
    test_loss = 0
    test_loss_cats = {}
    freq_cats = {}
    split_index_for_sents = {}
    deviations_from_expected = {"sent": [], "tree_type": [], "deviation": [], "binary_text": []}
    deviations_sents = []
    
    for test_sample in tqdm(test_loader):
        tree_col = "binary_tree_type" if full_dataset else "tree_type"
        sents = test_sample["sent"]
        sent_cat = test_sample[tree_col]
        bin_text = test_sample["binary_text"]
        actual_emb = test_sample["emb"]
        if not control_task:
            child_embs = test_sample["child_embs"]
        if control_task: # control task is to control for anisotropy, select random child vectors
            rand_parent = random.sample(range(0, len(test_loader.dataset)), min(BATCH_SIZE, actual_emb.shape[0])) 
            rand_left = random.sample(range(0, len(test_loader.dataset)), min(BATCH_SIZE, actual_emb.shape[0]))
            rand_right = random.sample(range(0, len(test_loader.dataset)), min(BATCH_SIZE, actual_emb.shape[0]))
            actual_emb = test_loader.dataset[rand_parent]["emb"]
            left_emb_rand = test_loader.dataset[rand_left]["child_embs"][:, 0, :]
            right_emb_rand = test_loader.dataset[rand_right]["child_embs"][:, 1, :]
            child_embs = torch.stack((left_emb_rand, right_emb_rand), dim=1)

        if probe_type == "add":
            if torch.cuda.is_available:
                composition_fn = composition_functions.b_torch_add
            else:
                composition_fn = composition_functions.add
        elif probe_type == "mult":
            if torch.cuda.is_available:
                composition_fn = composition_functions.b_torch_mult
            else:
                composition_fn = composition_functions.mult
        elif probe_type == "w1":
            composition_fn = composition_functions.b_w1
        elif probe_type == "w2":
            composition_fn = composition_functions.b_w2
        elif probe_type == "oracle":
            composition_fn_1 = composition_functions.b_w1
            composition_fn_2 = composition_functions.b_w2
        else:
            model.eval()
            composition_fn = model
        
        with torch.no_grad():
            if torch.cuda.is_available():
                actual_emb = actual_emb.to("cuda")
                child_embs = child_embs.to("cuda")

            if probe_type == "oracle":
                child_comp_emb_1 = composition_fn_1(child_embs)
                child_comp_emb_2 = composition_fn_2(child_embs)
                loss1 = loss_fn(actual_emb, child_comp_emb_1)
                loss2 = loss_fn(actual_emb, child_comp_emb_2)
                loss = min(loss1.item(), loss2.item())
                test_loss += loss
            else:
                try:
                    child_comp_emb = composition_fn(child_embs)
                    loss = loss_fn(actual_emb, child_comp_emb)
                    sum_loss = torch.sum(loss)
                    test_loss += sum_loss.item()
                except:
                    pdb.set_trace()

                if record_deviation:
                    for i, sent in enumerate(test_sample["sent"]):
                        deviations_from_expected["sent"].append(sent)
                        deviations_from_expected["binary_text"].append(bin_text[i])
                        deviations_from_expected["tree_type"].append(sent_cat[i])
                        if loss[i].item() < 0: # can sometimes happen with underflow?
                            item_loss = 0.0
                        else:
                            item_loss = loss[i].item()
                        deviations_from_expected["deviation"].append(item_loss)

        for i, s_c in enumerate(sent_cat):
            if s_c not in test_loss_cats:
                test_loss_cats[s_c] = loss[i].item()
                freq_cats[s_c] = 1
            else:
                test_loss_cats[s_c] += loss[i].item()
                freq_cats[s_c] += 1
    
    return test_loss, test_loss_cats, freq_cats, deviations_from_expected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("probe_type", help="probe type (predict parent vec from child vecs)", choices=["add", "mult", "w1", "w2", "linear", "mlp", "tpdn", "oracle", "affine"])
    parser.add_argument("--emb_type", help="type of embedding to approximate", choices=["cls", "avg"], default="cls")
    parser.add_argument("--model", help="type of model to examine", choices=["bert", "roberta", "deberta", "gpt2"], default="bert")
    parser.add_argument("--full", help="use the full treebank (default only 10%)", action="store_true")
    parser.add_argument("--use_binary", help="use binary parse trees (default is n-ary trees)", action="store_true")
    parser.add_argument("--use_control_task", help="evaluate probe on the control task (default is on the true task)", action="store_true")
    parser.add_argument("--use_trained", help="use pretrained probes (to evaluate other layers)", action="store_true")
    parser.add_argument("-l", "--layers", help="Layer to examine", type=int, default=12, choices=[i for i in range(1, 13)])
    args = parser.parse_args()
    vec_type = "CLS" if args.emb_type == "cls" else "avg"

    fit_composition(args.probe_type, vec_type=vec_type, full=args.full, binary_trees=args.use_binary, control_task=args.use_control_task, use_trained=args.use_trained, lm_type=args.model)
