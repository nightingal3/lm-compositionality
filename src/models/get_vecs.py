from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, BertTokenizer, BertModel, RobertaModel, RobertaTokenizer, DebertaModel, DebertaTokenizer
from typing import List
import pdb
import torch
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def model_init(model_string: str, cuda: bool, output_attentions=False, fast=False):
    if model_string == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    elif model_string == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True)
    elif model_string == "deberta":
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        model = DebertaModel.from_pretrained("microsoft/deberta-base", output_hidden_states=True)
    elif model_string.startswith("gpt2"):
        if fast:
            tokenizer = AutoTokenizer.from_pretrained(model_string)
            model = GPT2LMHeadModel.from_pretrained(model_string)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_string) 
            model = GPT2LMHeadModel.from_pretrained(model_string)
    elif model_string.startswith("EleutherAI/gpt-neo"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string, output_attentions=output_attentions)
        model = GPTNeoForCausalLM.from_pretrained(model_string, output_attentions=output_attentions)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    model.eval()
    if cuda:
        model.to('cuda')
    return model, tokenizer

def get_one_vec(phrase, model, tokenizer, emb_type: str = "CLS", cuda: bool = False):
    encoded_phrase = tokenizer.encode(phrase, return_tensors="pt")
    if cuda:
        encoded_phrase = encoded_phrase.to("cuda")

    outputs = model(encoded_phrase)
    if emb_type == "CLS":
        emb_phrase = outputs[0][:, 0, :]
    elif emb_type == "avg":
        token_embeddings = outputs["last_hidden_state"]
        emb_phrase = token_embeddings.squeeze(0).mean(dim=0)

    return emb_phrase
    

def get_vecs(phrase: str, model, tokenizer, cos_sim_func, cuda: bool = False, segmentation: List = [], type: str = "cls") -> tuple:
    encoded_phrase = tokenizer.encode(phrase, return_tensors="pt").to("cuda")
    if segmentation == []:
        segmentation = phrase.split()
    encoded_children = [tokenizer.encode(w, return_tensors="pt").to("cuda") for w in segmentation]

    outputs_phrase = model(encoded_phrase)

    if type == "cls":
        emb_phrase = outputs_phrase[0][:, 0, :]

    outputs_children = [model(child_enc)[0][:, 0, :] for child_enc in encoded_children]
    child_sum = sum(outputs_children)
    #child_sum = outputs_children[0]
    return 1 - cos_sim_func(emb_phrase, child_sum).item()

def get_bert_correlation_human_judgment(bert_scores, human_scores) -> float:
    return pearsonr(bert_scores, list(human_scores))

if __name__ == "__main__":
    model, tokenizer = model_init("bert", True)
    cos = torch.nn.CosineSimilarity(dim=1)
    idioms_df = pd.read_csv("./data/idioms/eng.csv")

    columns = ['#word', 'Word1_mean', 'Word1_std', 'Word2_mean', 'Word2_std', 'Cpd_mean', 'Cpd_std', 'mean1*mean2']
    compounds_df = pd.read_csv("./data/ijcnlp_compositionality_data/MeanAndDeviations.clean.txt", delimiter="\t|\s+", names=columns, header=None, skiprows=1)
    compounds_df = compounds_df.reset_index(level=0)
    compounds_df["compound_phrase"] = compounds_df["index"] + " " + compounds_df["#word"]
    input_data, comp_human_judgements = compounds_df[["compound_phrase", "index", "#word"]], compounds_df["Cpd_mean"]

    
    #for row in idioms_df.to_dict(orient="records"):
        #idiom = row["idiom"]
        #print(idiom)
        #sim = get_vecs(idiom, model, tokenizer, cos)
        #print(sim)

    #cos_sim = []
    #for row in compounds_df.to_dict(orient="records"):
        #compound = row["compound_phrase"]
        #cos_sim.append(get_vecs(compound, model, tokenizer, cos))

    #print("ADD: ", get_bert_correlation_human_judgment(cos_sim, comp_human_judgements))
    nums = []
    num_words = []
    cos_sim = []
    with open("./data/num_list_en.csv", "r") as f:
        for row in f.readlines():
            row = row.strip()
            i, word = row.split(",")
            word = word.strip()
            is_comp = False
            print(i)
            if int(i) > 10000:
                break
            if len(word.split()) == 1:
                continue
            if len(word.split()) == 2 and (word.split()[1] == "hundred" or word.split()[1] == "thousand"):
                continue
            
            if word.split()[1] == "hundred":
                split_word = word.split()
                segmentation = [" ".join(split_word[:2]), split_word[2], " ".join(split_word[3:])]
                is_comp = True
            if word.split()[1] == "thousand":
                split_word = word.split()
                if "hundred" in split_word:
                    hund_ind = split_word.index("hundred")
                    if hund_ind == len(split_word) - 1:
                        segmentation = [" ".join(split_word[:2]), " ".join(split_word[2:])]
                    else:
                        segmentation = [" ".join(split_word[:2]), " ".join(split_word[2:hund_ind + 1]), split_word[hund_ind + 1], " ".join(split_word[hund_ind + 2:])]
                else:
                    segmentation = [" ".join(split_word[:2]), split_word[2], " ".join(split_word[3:])]
                print(segmentation)
            nums.append(int(i))
            num_words.append(word)
            if is_comp:
                cos_sim.append(get_vecs(word, model, tokenizer, cos, segmentation=segmentation))
            else:
                cos_sim.append(get_vecs(word, model, tokenizer, cos))
    print(cos_sim)
    plt.plot(nums, cos_sim)
    plt.savefig("./numbers.png")
    df = pd.DataFrame({"num": nums, "word": num_words, "cos_dist": cos_sim}).sort_values(by="cos_dist")
    df.to_csv("number_cos_dist.csv", index=False)

    
