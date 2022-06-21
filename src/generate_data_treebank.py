# Basically prototyping what we're going to do on a few examples from the corpus
import nltk
nltk.download('treebank')
import nltk.corpus
from nltk import Tree
from nltk.corpus.reader import CategorizedBracketParseCorpusReader
from random import sample
from typing import List, Tuple
from scipy.spatial.distance import cosine
from io import TextIOWrapper
import pandas as pd
from pprint import pprint
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import pdb
import spacy
from functools import reduce
from copy import deepcopy
import re

from src.models.get_vecs import model_init

NULL_VALUES = ["0", "NIL", "*", "*u*", "*ich*", "*exp*", "*rnr*", "*ppa*", "*?*", "-lrb-", "-rrb-"] + [f"*t*-{i}" for i in range(1, 1000)] + [f"*-{i}" for i in range(1, 1000)] + [f"*exp*-{i}" for i in range(1, 1000)] + [f"*ich*-{i}" for i in range(1, 1000)] + [f"*rnr*-{i}" for i in range(1, 1000)] + [f"*ppa*-{i}" for i in range(1, 1000)]
NULL_VALUES = set(NULL_VALUES)

def get_full_sents(selected_files, treebank):
  sents = []

  for i in range(len(selected_files)):
    sents.extend([s for s in treebank.parsed_sents(selected_files[i])])
  
  return sents

# get all the strings associated with the different subtrees, organized by
# depth in the tree and position. Also return a list of leaves for convenience.
def get_subtree_strings(tree: nltk.tree.Tree, return_tree_type: bool = False) -> Tuple[dict, List]:
  strings_by_loc = {}
  leaf_set = []
  for postn in tree.treepositions():
    try:
      if "-NONE-" in tree[postn].label():
        continue
      leaves = tree[postn].leaves()
      parse_string = " ".join(str(tree[postn]).split()) 
      try:
        copy = deepcopy(tree[postn])
        copy.chomsky_normal_form()
        binary_parse_string = " ".join([w for w in str(copy).split()])
      except:
        print("fail")
      text = " ".join([w.lower() for w in leaves if w.lower() not in NULL_VALUES])
      is_leaf = len(leaves) == 1
      if return_tree_type:
        tree_type = str(tree[postn].productions()[0])
        binary_tree_type = str(copy.productions()[0])
        if is_leaf: 
          tree_type = tree_type[:tree_type.index("->") - 1]
      if is_leaf:
        leaf_set.append(text)
      depth = len(postn)
      if depth not in strings_by_loc:
        strings_by_loc[depth] = {}
      if depth == 0:
        postn = "ROOT" #the bracket character is weird

      if return_tree_type:
        strings_by_loc[depth][postn] = (tree_type, binary_tree_type, text, parse_string, binary_parse_string)
      else:
        strings_by_loc[depth][postn] = text
    except:
      continue

  return strings_by_loc, leaf_set


# from https://stackoverflow.com/questions/44742809/how-to-get-a-binary-parse-in-python.
# will be wrong in some cases
def _binarize(tree):
    """
    Recursively turn a tree into a binary tree.
    """
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        return binarize(tree[0])
    else:
        label = tree.label()
        return reduce(lambda x, y: Tree(label, (binarize(x), binarize(y))), tree)

def get_subtree_strings_flat(tree: nltk.tree.Tree) -> List:
  strings = set()
  for postn in tree.treepositions():
    try:
      leaves = tree[postn].leaves()
      text = " ".join([w.lower() for w in leaves if w.lower() not in NULL_VALUES])
      strings.add(text)
    except:
      continue
    
  return list(strings)

def get_df_from_full_sentences(selected_files, treebank_obj) -> Tuple[pd.DataFrame, int]:
  MAX_LENGTH = 0
  NER = spacy.load("en_core_web_sm")
  idioms = set(pd.read_csv("./data/idioms/eng.csv")["idiom"])

  all_sentences = get_full_sents(selected_files, treebank_obj)
  sentence_col = []
  subsentence_col = []
  sublength_col = []
  sentence_len_col = []
  depth_col = []
  tree_index_col = []
  tree_type_col = []
  binary_tree_type_col = []
  named_entities = []
  named_entity_types = []
  is_named_entity = []
  is_idiom = []
  parse_strings = []
  binary_parse_strings = []

  for sent in all_sentences:
    sentence_positions, leaves = get_subtree_strings(sent, return_tree_type=True)
    full_len = len(leaves)
    if full_len > MAX_LENGTH:
      MAX_LENGTH = full_len
    for depth in sentence_positions:
      for tree_ind in sentence_positions[depth]:
        tree_type, binary_tree_type, text, parse_string, binary_parse_string = sentence_positions[depth][tree_ind]
        sentence_col.append(sentence_positions[0]['ROOT'][2])
        subsentence_col.append(text)
        sentence_len_col.append(full_len)
        sublength_col.append(len(text.split()))
        depth_col.append(depth)
        tree_index_col.append(tree_ind)
        tree_type_col.append(tree_type)
        binary_tree_type_col.append(binary_tree_type)
        binary_parse_strings.append(binary_parse_string)
        parse_strings.append(parse_string)
        ner_tagged_text = NER(text)
        named_entities.append(ner_tagged_text.ents)
        named_entity_types.append([word.label_ for word in ner_tagged_text.ents])
        if len(ner_tagged_text.ents) == 1 and ner_tagged_text.ents[0].text == text:
          is_named_entity.append(True)
        else:
          is_named_entity.append(False)
        
        is_idiom.append(text in idioms)
        if text in idioms:
          print("idiom!!!!", text)
  
  col_dict = {"full_sent": sentence_col, "full_length": sentence_len_col,
              "sent": subsentence_col, "sublength": sublength_col,
              "depth": depth_col, "tree_ind": tree_index_col,
              "tree_type": tree_type_col, "parse": parse_strings, "binary_parse": binary_parse_strings, "binary_tree_type": binary_tree_type_col,
              "named_ents": named_entities, "named_ent_types": named_entity_types, "is_named_entity": is_named_entity, "is_idiom": is_idiom}

  return pd.DataFrame(col_dict), MAX_LENGTH

def get_model_embeddings(model, dataloader, comparison_type: str = "CLS", model_type: str = "encoder", cuda: bool = False, layer: int = 12):
  embeddings = None
  i = 0
  with torch.no_grad():
    for batch in dataloader:
      print(f"predicting batch {i}", "out of", len(dataloader))
      i += 1
      b_input_ids, b_attn_masks = batch
      outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attn_masks)
      if comparison_type == "CLS":
        if layer == 12:
          if model_type == "encoder":
            embedding = outputs["last_hidden_state"][:,0,:].detach().cpu().numpy()
          else:
            embedding = outputs["last_hidden_state"][:, -1, :].detach().cpu().numpy()
        else:
          hidden = outputs["hidden_states"][layer]
          if model_type == "encoder":
            embedding = hidden[:, 0, :].detach().cpu().numpy()
          else:
            embedding = hidden[:, -1, :].detach().cpu().numpy()
      elif comparison_type == "avg":
        if layer == 12:
          token_embeddings = outputs["last_hidden_state"]
          token_embeddings[b_attn_masks == 0] = 0 # get rid of padding vectors 
          embedding = (token_embeddings.sum(axis=1) / b_attn_masks.sum(axis=1).unsqueeze(-1)).detach().cpu().numpy()
        else:
          token_embeddings = outputs["hidden_states"][layer]
          token_embeddings[b_attn_masks == 0] = 0
          embedding = (token_embeddings.sum(axis=1) / b_attn_masks.sum(axis=1).unsqueeze(-1)).detach().cpu().numpy()

      elif comparison_type == "max":
        token_embeddings = outputs["last_hidden_state"]
        embedding, _ = torch.max((token_embeddings * b_attn_masks.unsqueeze(-1)), axis=1)
        embedding = embedding.detach().cpu().numpy()
      assert embedding.shape[0] == len(b_input_ids) and embedding.shape[1] == 768
      if embeddings is None:
        embeddings = embedding
      else:
        embeddings = np.vstack((embeddings, embedding))
    #pdb.set_trace()
    #embeddings.append(embedding)
  
  #embeddings = np.array(embeddings)
  return embeddings

def get_one_bert_emb(model, tokenizer, phrase, emb_type: str = "CLS", cuda: bool = False):
  raise NotImplementedError

def get_bert_embeddings_with_context(dataloader):
  embeddings = None
  for i, batch in enumerate(dataloader):
    # keep track of which vector positions we need to get
    b_input_ids, b_attn_masks, b_spans = batch
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attn_masks)
    token_embeddings = outputs["last_hidden_state"]

# distance between subsentences?
def get_root_embs(df: pd.DataFrame) -> pd.DataFrame:
  root_sents = df.loc[df["tree_ind"] == "ROOT"]
  return root_sents[["full_sent", "emb"]]

def get_dist_from_root(df: pd.DataFrame, root_embs: pd.DataFrame):
  subtree_dists = []
  for i, row in df.iterrows():
    try:
      root_emb = root_embs.loc[root_embs["full_sent"] == row["full_sent"]]["emb"].item()
    except: # duplicate sentence (?)
      root_emb = root_embs.loc[root_embs["full_sent"] == row["full_sent"]]["emb"].values[0]
    dist = cosine(row["emb"], root_emb)
    subtree_dists.append(dist)
  return subtree_dists

def unique_phrase_df(sents_list: List, out_filename: str) -> None:
  unique_phrases = set(sents_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process treebank files for subsentences and return records including BERT embeddings, tree types, and sentence positions.")
    parser.add_argument("--model", help="model to extract embeddings for", choices=["bert", "roberta", "deberta", "gpt2"], default="bert")
    parser.add_argument("-i", dest="selection_num", help="process number (split into 10)")
    parser.add_argument("--emb_type", help="embedding type to use", choices=["CLS", "avg", "max"], default="CLS")
    parser.add_argument("--treebank_path", help="path to treebank files. Defaults to NLTK treebank sample if not supplied.", action="store_true")
    parser.add_argument("--cuda", help="use gpu", action="store_true")
    parser.add_argument("--layer", help="layer embeddings to collect", choices=[i for i in range(1, 13)], default=12, type=int)
    args = parser.parse_args()
            
    if args.treebank_path: # brown: 192 files, wsj: 2313
      treebank = CategorizedBracketParseCorpusReader(r"/projects/tir1/corpora/treebank_3/parsed/mrg/", r'(wsj/\d\d/wsj_\d\d\d\d|brown/c[a-z]/c[a-z]\d\d).mrg', cat_file='allcats.txt', tagset='wsj')
    else:
      treebank = nltk.corpus.treebank
    selected_files = [fileid for i, fileid in enumerate(list(treebank.fileids())) if (i % 10) == int(args.selection_num)]
    df, _ = get_df_from_full_sentences(selected_files, treebank)
    df = df.loc[df["sublength"] >= 2] # drop trivial trees

    model, tokenizer = model_init(args.model, cuda=args.cuda)
    model_type = "gpt" if args.model == "gpt2" else "encoder"

    test_samples = df["sent"].tolist()
    encoded_data_val = tokenizer(test_samples, 
                                add_special_tokens=True, 
                                return_attention_mask=True, 
                                padding='longest',
                                truncation=True,
                                max_length=512, 
                                return_tensors='pt')
    if args.cuda:
      model.to("cuda")
      encoded_data_val.to("cuda")

    input_ids = encoded_data_val['input_ids']
    attention_masks = encoded_data_val['attention_mask']
    
    assert len(input_ids) == len(attention_masks) == len(df)

    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=16)
    
    embeddings = get_model_embeddings(model, dataloader, comparison_type=args.emb_type, model_type=model_type, cuda=args.cuda, layer=args.layer)
    df["emb"] = embeddings.tolist()
    np.save(f"./data/embs/{args.model}_{args.emb_type}_{args.selection_num}_full_{args.treebank_path}_layer_{args.layer}.npy", embeddings)

    #root_to_emb = get_root_embs(df)
    #subtree_dists = get_dist_from_root(df, root_to_emb)
    #df["dist_from_root"] = subtree_dists
    #depth_1_subtree_rows = df.loc[df["depth"] == 1]
    #depth_1_subtree_dists = depth_1_subtree_rows["dist_from_root"].tolist()
    df.to_csv(f"./data/raw_tree_data/tree_data_{args.model}_{args.selection_num}_{args.emb_type}_{args.treebank_path}_layer_{args.layer}.csv")
