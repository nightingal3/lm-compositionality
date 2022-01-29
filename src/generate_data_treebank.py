# Basically prototyping what we're going to do on a few examples from the corpus
import nltk
nltk.download('treebank')
import nltk.corpus
from nltk import Tree
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

MAX_LENGTH = 123 # have to determine this manually

def get_full_sents(selected_files):
  sents = []

  for i in range(len(selected_files)):
    sents.extend([s for s in treebank.parsed_sents(selected_files[i])])
  
  return sents

# get all the strings associated with the different subtrees, organized by
# depth in the tree and position. Also return a list of leaves for convenience.
def get_subtree_strings(tree: nltk.tree.Tree, return_tree_type: bool = False) -> Tuple[dict, List]:
  strings_by_loc = {}
  leaf_set = set()
  for postn in tree.treepositions():
    try:
      if "-NONE-" in tree[postn].label():
        continue
      leaves = tree[postn].leaves()
      text = " ".join([w.lower() for w in leaves if w != "*-1"])
      is_leaf = len(leaves) == 1
      if return_tree_type:
        tree_type = str(tree[postn].productions()[0])
        if is_leaf: 
          tree_type = tree_type[:tree_type.index("->") - 1]
      if is_leaf:
        leaf_set.add(text)
      depth = len(postn)
      if depth not in strings_by_loc:
        strings_by_loc[depth] = {}
      if depth == 0:
        postn = "ROOT" #the bracket character is weird

      if return_tree_type:
        strings_by_loc[depth][postn] = (tree_type, text)
      else:
        strings_by_loc[depth][postn] = text
    except:
      continue
    
  return strings_by_loc, list(leaf_set)

def get_subtree_strings_flat(tree: nltk.tree.Tree) -> List:
  strings = set()
  for postn in tree.treepositions():
    try:
      leaves = tree[postn].leaves()
      text = " ".join([w.lower() for w in leaves if w != "*-1"])
      strings.add(text)
    except:
      continue
    
  return list(strings)

def get_df_from_full_sentences(selected_files) -> Tuple[pd.DataFrame, int]:
  MAX_LENGTH = 0
  NER = spacy.load("en_core_web_sm")

  all_sentences = get_full_sents(selected_files)
  sentence_col = []
  subsentence_col = []
  sublength_col = []
  sentence_len_col = []
  depth_col = []
  tree_index_col = []
  tree_type_col = []
  named_entities = []
  named_entity_types = []

  for sent in all_sentences:
    sentence_positions, leaves = get_subtree_strings(sent, return_tree_type=True)
    full_len = len(leaves)
    if full_len > MAX_LENGTH:
      MAX_LENGTH = full_len
    for depth in sentence_positions:
      for tree_ind in sentence_positions[depth]:
        tree_type, text = sentence_positions[depth][tree_ind]
        sentence_col.append(sentence_positions[0]['ROOT'][1])
        subsentence_col.append(text)
        sentence_len_col.append(full_len)
        sublength_col.append(len(text))
        depth_col.append(depth)
        tree_index_col.append(tree_ind)
        tree_type_col.append(tree_type)

        ner_tagged_text = NER(text)
        named_entities.append(ner_tagged_text.ents)
        named_entity_types.append([word.label_ for word in ner_tagged_text.ents])
  
  col_dict = {"full_sent": sentence_col, "full_length": sentence_len_col,
              "sent": subsentence_col, "sublength": sublength_col,
              "depth": depth_col, "tree_ind": tree_index_col,
              "tree_type": tree_type_col, "named_ents": named_entities, "named_ent_types": named_entity_types}

  return pd.DataFrame(col_dict), MAX_LENGTH

def get_bert_embeddings(dataloader, comparison_type: str = "CLS"):
  embeddings = None
  for i, batch in enumerate(dataloader):
    print(f"predicting batch {i}", "out of", len(dataloader))
    b_input_ids, b_attn_masks = batch
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attn_masks)
    if comparison_type == "CLS":
      embedding = outputs[0][:,0,:].detach().numpy()
    elif comparison_type == "avg":
      token_embeddings = outputs["last_hidden_state"]
      embedding = (token_embeddings.sum(axis=1) / b_attn_masks.sum(axis=-1).unsqueeze(-1)).detach().numpy()
    elif comparison_type == "max":
      token_embeddings = outputs["last_hidden_state"]
      embedding, _ = torch.max((token_embeddings * b_attn_masks.unsqueeze(-1)), axis=1)
      embedding = embedding.cpu().detach().numpy()

    assert embedding.shape[0] == len(b_input_ids) and embedding.shape[1] == 768
    if embeddings is None:
      embeddings = embedding
    else:
      embeddings = np.vstack((embeddings, embedding))
      
    
  return embeddings

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
    parser.add_argument("-i", dest="selection_num", help="process number (split into 10)")
    parser.add_argument("--emb_type", help="embedding type to use", choices=["CLS", "avg", "max"], default="CLS")
    args = parser.parse_args()

    treebank = nltk.corpus.treebank
    selected_files = [fileid for i, fileid in enumerate(list(treebank.fileids())) if (i % 10) == int(args.selection_num)]
    df, _ = get_df_from_full_sentences(selected_files)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained('bert-base-uncased',
    output_attentions = False, 
    output_hidden_states = True)
    test_samples = df["sent"].tolist()
    encoded_data_val = tokenizer.batch_encode_plus(test_samples, 
                                               add_special_tokens=True, 
                                               return_attention_mask=True, 
                                               padding='longest',
                                               truncation=True,
                                               max_length=MAX_LENGTH, 
                                               return_tensors='pt')
    input_ids = encoded_data_val['input_ids']
    attention_masks = encoded_data_val['attention_mask']
    
    NER = spacy.load("en_core_web_sm")

    assert len(input_ids) == len(attention_masks) == len(df)

    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)

    embeddings = get_bert_embeddings(dataloader, comparison_type=args.emb_type)
    pdb.set_trace()
    df["emb"] = embeddings.tolist()
    np.save(f"./data/embs/{args.emb_type}_{args.selection_num}.npy", embeddings)

    #root_to_emb = get_root_embs(df)
    #subtree_dists = get_dist_from_root(df, root_to_emb)
    #df["dist_from_root"] = subtree_dists
    #depth_1_subtree_rows = df.loc[df["depth"] == 1]
    #depth_1_subtree_dists = depth_1_subtree_rows["dist_from_root"].tolist()
    df.to_csv(f"./tree_data_{args.selection_num}_{args.emb_type}.csv")
