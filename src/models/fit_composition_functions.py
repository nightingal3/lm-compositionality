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

def fit_composition(probe_type: str = "add", vec_type: str = "CLS", loss_type: str = "cosine", rand_seed: int = 42, full: bool = False, binary_trees: bool = False, decomp: bool = False) -> None:
    seed_everything(rand_seed)
    log_path =  f"./data/{probe_type}_{vec_type}_results/binary_trees/" if binary_trees else f"./data/{probe_type}_{vec_type}_results/"

    if binary_trees:
        #lm_model, lm_tokenizer = model_init("bert", cuda=torch.cuda.is_available())
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
        pdb.set_trace()
        #dataset_full = load_dataset("csv", data_files=[f"data/tree_data_{i}_avg_full_True_proto_False.csv" for i in range(10)], split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
        #if vec_type == "avg":
            #dataset_full = load_dataset("csv", data_files=[f"tree_data_{i}_{vec_type}_True.csv" for i in range(10)], split=[
            #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
            #])
        #else:
            #dataset_full = load_dataset("csv", data_files=[f"tree_data_{i}_{vec_type}_True.csv" for i in range(10)], split=[
            #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
            #])
        unsliced_dataset = Dataset.from_pandas(data_df)
        kfold = KFold(n_splits=10, shuffle=False)
        full_dataset = kfold.split(unsliced_dataset)
        
    else:
        pdb.set_trace()
        data_df = pd.read_csv(f"./data/small_test/{vec_type}_sample.csv")
        #dataset_full = load_dataset("csv", data_files=f"./data/small_test/{vec_type}_sample.csv", split=[
        #f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
        #])
        unsliced_dataset = Dataset.from_pandas(data_df, split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)])
        # loading datasets using load_dataset and split doesn't work for some reason....
        kfold = KFold(n_splits=10, shuffle=False)
        full_dataset = kfold.split(unsliced_dataset)
        pdb.set_trace()

    if loss_type == "cosine":
        loss_fn = lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=0)
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()

    overall_loss = {}
    overall_loss_cats = {}
    total_test_loss_cats = {}
    partition_len = {}
    freq_cats_overall = {}
    for i, partition in enumerate(dataset_full):
        print(f"=== PARTITION {i} ===")
        if probe_types[probe_type] == "par":
            test_data = partition
            train_data = concatenate_datasets([dataset_full[j] for j in range(len(dataset_full)) if j != i])
            train_data = train_data.shuffle(seed=rand_seed)
        else:
            train_data = []
            test_data = partition

        print("~TRAIN~")
        if probe_types[probe_type] == "par":
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
            for sample in tqdm(train_data):
                actual_emb = torch.FloatTensor(ast.literal_eval(sample["emb"]))
                child_id_lst = ast.literal_eval(sample["child_ids"])

                if torch.cuda.is_available():
                    actual_emb = actual_emb.to("cuda")


                #TODO: change this to get from child IDs (currently hash is broken)
                #child_embs = torch.FloatTensor(get_child_embs_from_id(data_df, child_id_lst))
                if binary_trees:
                    #binary_text = ast.literal_eval(sample["binary_text"])
                    #if len(binary_text) < 2:
                        #continue
                    #left_text, right_text = binary_text
                    #left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                    #right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                    #child_embs = torch.stack([left_emb, right_emb]).squeeze(1)
                    pdb.set_trace()
                else:
                    child_embs = torch.FloatTensor(get_child_embs(data_df, sample["full_sent"], sample["tree_ind"], sample["depth"])[0])

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

            if probe_types[probe_type] == "par":
                Path(log_path).mkdir(parents=True, exist_ok=True)
                model_name = f"model_{i}_bintree" if binary_trees else f"model_{i}"  
                torch.save(model.state_dict(), f"{log_path}/{model_name}.pt")

        print("~TEST~")
        test_loss = 0
        test_loss_cats = {}
        freq_cats = {}
        non_leaves = 0
        for test_sample in tqdm(test_data):
            sent_cat = test_sample["tree_type"]
            actual_emb = torch.FloatTensor(ast.literal_eval(test_sample["emb"]))
            child_ids = ast.literal_eval(test_sample["child_ids"])
            
            if binary_trees:
                binary_text = ast.literal_eval(test_sample["binary_text"])
                if len(binary_text) < 2:
                    continue
                left_text, right_text = binary_text
                left_emb = get_one_vec(left_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                right_emb = get_one_vec(right_text, lm_model, lm_tokenizer, emb_type=vec_type, cuda=torch.cuda.is_available())
                child_embs = torch.stack([left_emb, right_emb]).squeeze(1)
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
            else:
                model.eval()
                composition_fn = model
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    actual_emb = actual_emb.to("cuda")
                    child_embs = child_embs.to("cuda")
                child_comp_emb = composition_fn(child_embs)
                loss = loss_fn(actual_emb, child_comp_emb)
                test_loss += loss.item()
            if sent_cat not in test_loss_cats:
                test_loss_cats[sent_cat] = loss.item()
                freq_cats[sent_cat] = 1
            else:
                test_loss_cats[sent_cat] += loss.item()
                freq_cats[sent_cat] += 1
        
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("probe_type", help="probe type (predict parent vec from child vecs)", choices=["add", "mult", "w1", "w2", "linear", "mlp", "tpdn"])
    parser.add_argument("--emb_type", help="type of embedding to approximate", choices=["cls", "avg"], default="cls")
    parser.add_argument("--full", help="use the full treebank (default only 10%)", action="store_true")
    parser.add_argument("--use_binary", help="use binary parse trees (default is n-ary trees)", action="store_true")
    parser.add_argument("--decomp", help="train decompositional probes instead (default is compositional)", action="store_true")

    args = parser.parse_args()
    vec_type = "CLS" if args.emb_type == "cls" else "avg"

    if args.decomp:
        raise NotImplementedError
    fit_composition(args.probe_type, vec_type=vec_type, full=args.full, decomp=args.decomp, binary_trees=args.use_binary)