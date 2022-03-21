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
    "mlp": "par",
    "tpdn": "par"
}

# for composition prediction, input size is number of children and output size is embedding size.
# for decomposition prediction, input size is embedding size and output size is number of children * embedding size
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, cuda: bool = False):
        super(LinearRegression, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = "cuda" if cuda else "cpu"
        self.W = torch.nn.Parameter(torch.randn((1, input_size), requires_grad=True, device=self.device))
        self.b = torch.nn.Parameter(torch.randn((1, output_size), requires_grad=True, device=self.device))

    def forward(self, x): 
        assert len(x) <= self.input_size
        if x.shape[0] < self.input_size:
            padding = torch.zeros((self.input_size - x.shape[0], self.output_size), device=self.device)
            x = torch.vstack([x, padding])
        pred = self.W.mm(x) + self.b
        return pred.squeeze(0)

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
        assert len(x) <= self.input_size
        if x.shape[0] < self.input_size:
            padding = torch.zeros((self.input_size - x.shape[0], self.output_size), device=self.device)
            x = torch.vstack([x, padding])
        hidden = self.hidden1(x.T)
        hidden_relu = self.relu(hidden)
        hidden_2 = self.hidden2(hidden_relu)
        hidden_2_relu = self.relu(hidden_2)
        out = self.output(hidden_2_relu)
        return out.T.squeeze(0)

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

def fit_composition(probe_type: str = "add", vec_type: str = "CLS", loss_type: str = "cosine", rand_seed: int = 42, full: bool = False, binary_trees: bool = False, control_task: bool = False) -> None:
    seed_everything(rand_seed)
    log_path =  f"./data/{probe_type}_{vec_type}_results/binary_trees/" if binary_trees else f"./data/{probe_type}_{vec_type}_results/"

    if binary_trees:
        lm_model, lm_tokenizer = model_init("bert", cuda=torch.cuda.is_available())
        lm_model.eval()
        vec_type_lower = vec_type.lower()
        if full:
            binary_vec_data = np.load(f"./data/binary_child_embs_{vec_type_lower}.npy", mmap_mode="r+")
            binary_mask_data = np.load(f"./data/binary_child_embs_{vec_type_lower}_mask.npy", mmap_mode="r+")
        else:
            binary_vec_data = np.load(f"./data/small_test/{vec_type}_bin_data_sample.npy", mmap_mode="r+")
            binary_mask_data = np.load(f"./data/small_test/{vec_type}_bin_mask_sample.npy", mmap_mode="r+")

    if full:
        log_path += "full/"
        data_df = pd.read_csv(f"./data/{vec_type}_all_full.csv")
        data_df["original_order"] = data_df.index
        #dataset_full = load_dataset("csv", data_files=[f"data/tree_data_{i}_avg_full_True_proto_False.csv" for i in range(10)], split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
        #if vec_type == "avg":
            #dataset_full = load_dataset("csv", data_files=[f"tree_data_{i}_{vec_type}_True.csv" for i in range(10)], split=[
            #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
            #])
        #else:
            #dataset_full = load_dataset("csv", data_files=[f"tree_data_{i}_{vec_type}_True.csv" for i in range(10)], split=[
            #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
            #])
        dataset_full = Dataset.from_pandas(data_df)
        kfold = KFold(n_splits=10, shuffle=False)
        dataset_full_inds = kfold.split(dataset_full)
        
    else:
        data_df = pd.read_csv(f"./data/small_test/{vec_type}_sample.csv")
        data_df["original_order"] = data_df.index
        data_df = data_df.dropna(subset=["sent"])

        #dataset_full = load_dataset("csv", data_files=f"./data/small_test/{vec_type}_sample.csv", split=[
        #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
        #])
        dataset_full = Dataset.from_pandas(data_df)
        # loading datasets using load_dataset and split doesn't work for some reason....
        kfold = KFold(n_splits=10, shuffle=False)
        dataset_full_inds = kfold.split(dataset_full)

    if loss_type == "cosine":
        loss_fn = lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=0)
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()

    overall_loss = {}
    overall_loss_cats = {}
    total_test_loss_cats = {}
    partition_len = {}
    freq_cats_overall = {}


    for i, (train_indices, test_indices) in enumerate(dataset_full_inds):
        print(f"=== PARTITION {i} ===")
        if probe_types[probe_type] == "par":
            #test_data = partition
            #train_data = concatenate_datasets([dataset_full[j] for j in range(len(dataset_full)) if j != i])
            #train_data = train_data.shuffle(seed=rand_seed)
            #test_data = dataset_full[test_indices]
            #train_data = dataset_full[train_indices]
            test_data = torch.utils.data.Subset(dataset_full, test_indices.tolist())
            train_data = torch.utils.data.Subset(dataset_full, train_indices.tolist())
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

        else:
            train_data = []
            test_data = dataset_full
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        print("~TRAIN~")
        if probe_types[probe_type] == "par":
            model = train(i, probe_type, train_loader, test_loader, binary_vec_data, loss_fn, log_path, binary_trees, record_training_curve=True)
        else:
            model = None

        print("~TEST~")
        test_loss = 0
        test_loss_cats = {}
        freq_cats = {}
        non_leaves = 0

        test_loss, test_loss_cats, freq_cats, non_leaves = test(model, probe_type, test_loader, binary_vec_data, loss_fn, log_path, binary_trees, control_task=control_task, lm_model=lm_model, lm_tokenizer=lm_tokenizer, vec_type=vec_type)
        
        if non_leaves == 0:
            print("No non-leaves?")
            continue
            
        avg_test_loss = test_loss / non_leaves
        avg_test_loss_cats = {cat: loss / freq_cats[cat] for cat, loss in test_loss_cats.items()}
        overall_loss[i] = avg_test_loss
        partition_len[i] = len(test_data)

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


def train(partition_num: int, probe_type: str, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, binary_vec_data: np.ndarray, loss_fn: Callable, log_path: str, binary_trees: bool, record_training_curve: bool = False):
    if probe_type == "linear":
        if binary_trees:
            model = LinearRegression(input_size=2, output_size=768, cuda=torch.cuda.is_available())
        else:
            model = LinearRegression(input_size=5, output_size=768, cuda=torch.cuda.is_available())
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

    all_children = []
    optimizer = torch.optim.Adam(model.parameters())
    training_milestones = [int(0.01 * len(train_loader)), int(0.05 * len(train_loader))] + [int(0.1 * i * len(train_loader)) for i in range(1, 10)]
    milestone_losses = []
    eval_loss_next = False
    for i, sample in enumerate(tqdm(train_loader)):
        if type(sample["emb"]) == "str":
            actual_emb = torch.FloatTensor(ast.literal_eval(sample["emb"]))
        else: # batch size of 1, maybe do batching later
            actual_emb = torch.FloatTensor(ast.literal_eval(sample["emb"][0]))

        if type(sample["child_ids"]) == "str":
            child_id_lst = ast.literal_eval(sample["child_ids"])
        else:
            child_id_lst = ast.literal_eval(sample["child_ids"][0])

        if torch.cuda.is_available():
            actual_emb = actual_emb.to("cuda")

        if i in training_milestones and record_training_curve:
            eval_loss_next = True

        #TODO: change this to get from child IDs (currently hash is broken)
        #child_embs = torch.FloatTensor(get_child_embs_from_id(data_df, child_id_lst))
        if binary_trees:
            binary_text = ast.literal_eval(sample["binary_text"][0])
            if len(binary_text) < 2:
                continue
            child_embs = torch.FloatTensor(binary_vec_data[sample["original_order"].item()])
        else:
            child_embs = torch.FloatTensor(get_child_embs(data_df, sample["full_sent"][0], sample["tree_ind"][0], sample["depth"][0])[0])

        if torch.cuda.is_available():
            child_embs = child_embs.to("cuda")
        all_children.append(len(child_embs))
        if len(child_embs) > 5: # idk what kind of sentence this would be
            continue
        if len(child_embs) == 0 or len(child_embs) == 1:
            continue
        optimizer.zero_grad()
        if probe_type == "tpdn":
            roles = torch.tensor(range(len(child_embs)), device="cuda", dtype=int)
            pred = model(child_embs, roles)
        else:
            pred = model(child_embs)
        loss = loss_fn(actual_emb, pred)
        loss.backward()
        optimizer.step()

        if eval_loss_next: #evaluate and save value when we reach a training milestone
            print("milestone")
            test_loss, test_loss_cats, freq_cats, non_leaves = test(model, probe_type, test_loader, binary_vec_data, loss_fn, log_path, binary_trees)
            model.train()
            avg_test_loss = test_loss / non_leaves
            milestone_losses.append(avg_test_loss)
            eval_loss_next = False

    if probe_types[probe_type] == "par":
        Path(log_path).mkdir(parents=True, exist_ok=True)
        model_name = f"model_{partition_num}_bintree" if binary_trees else f"model_{partition_num}"  
        torch.save(model.state_dict(), f"{log_path}/{model_name}.pt")

    if record_training_curve:
        losses_name = f"{log_path}/model_{partition_num}_losses.p"
        with open(losses_name, "wb") as loss_file:
            pickle.dump(milestone_losses, loss_file)

    return model

def test(model, probe_type: str, test_loader: torch.utils.data.DataLoader, binary_vec_data: np.ndarray, loss_fn: Callable, log_path: str, binary_trees: bool, control_task: bool = False, lm_model = None, lm_tokenizer = None, vec_type: str = "cls"):
    test_loss = 0
    test_loss_cats = {}
    freq_cats = {}
    non_leaves = 0
    split_index_for_sents = {}
    for test_sample in tqdm(test_loader):
        sent_cat = test_sample["tree_type"][0]
        actual_emb = torch.FloatTensor(ast.literal_eval(test_sample["emb"][0]))
        child_ids = ast.literal_eval(test_sample["child_ids"][0])
        
        if binary_trees:
            binary_text = ast.literal_eval(test_sample["binary_text"][0])
            if len(binary_text) < 2:
                continue

            if control_task:
                binary_text_split = test_sample["sent"][0].split()
                if test_sample["sent"][0] in split_index_for_sents:
                    rand_part_ind = test_sample["sent"]
                else:
                    rand_part_ind = random.randint(0, len(binary_text_split) - 1)
                test_sample["sent"][0] = rand_part_ind
                left_text, right_text = " ".join(binary_text_split[:rand_part_ind]), " ".join(binary_text_split[rand_part_ind:])
                left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                child_embs = torch.stack([left_emb, right_emb]).squeeze(1)
            else:
                child_embs = torch.FloatTensor(binary_vec_data[test_sample["original_order"].item()])

            #left_text, right_text = binary_text
            #left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
            #right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
            #child_embs = torch.stack([left_emb, right_emb]).squeeze(1)
            #pdb.set_trace()
        else:
            child_embs = torch.FloatTensor(get_child_embs(data_df, test_sample["full_sent"], test_sample["tree_ind"], test_sample["depth"])[0])
        if len(child_embs) == 0 or len(child_embs) == 1:
            continue
        if (probe_type == "linear" or probe_type == "mlp") and len(child_embs) > 5:
            continue

        non_leaves += 1
        if probe_type == "add":
            if torch.cuda.is_available:
                composition_fn = composition_functions.torch_add
            else:
                composition_fn = composition_functions.add
        elif probe_type == "mult":
            if torch.cuda.is_available:
                composition_fn = composition_functions.torch_mult
            else:
                composition_fn = composition_functions.mult
        elif probe_type == "w1":
            composition_fn = lambda arr: arr[0]
        elif probe_type == "w2":
            composition_fn = lambda arr: arr[-1]
        elif probe_type == "oracle":
            composition_fn_1 = lambda arr: arr[0]
            composition_fn_2 = lambda arr: arr[-1]
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
                child_comp_emb = composition_fn(child_embs)
                loss = loss_fn(actual_emb, child_comp_emb)
                test_loss += loss.item()

        if sent_cat not in test_loss_cats:
            test_loss_cats[sent_cat] = loss.item()
            freq_cats[sent_cat] = 1
        else:
            test_loss_cats[sent_cat] += loss.item()
            freq_cats[sent_cat] += 1
        
    return test_loss, test_loss_cats, freq_cats, non_leaves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("probe_type", help="probe type (predict parent vec from child vecs)", choices=["add", "mult", "w1", "w2", "linear", "mlp", "tpdn", "oracle"])
    parser.add_argument("--emb_type", help="type of embedding to approximate", choices=["cls", "avg"], default="cls")
    parser.add_argument("--full", help="use the full treebank (default only 10%)", action="store_true")
    parser.add_argument("--use_binary", help="use binary parse trees (default is n-ary trees)", action="store_true")
    parser.add_argument("--decomp", help="train decompositional probes instead (default is compositional)", action="store_true")
    parser.add_argument("--use_control_task", help="evaluate probe on the control task (default is on the true task)", action="store_true")
    args = parser.parse_args()
    vec_type = "CLS" if args.emb_type == "cls" else "avg"

    if args.decomp:
        raise NotImplementedError
    fit_composition(args.probe_type, vec_type=vec_type, full=args.full, binary_trees=args.use_binary, control_task=args.use_control_task)