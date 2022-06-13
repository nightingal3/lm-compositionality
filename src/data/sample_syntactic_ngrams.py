import pandas as pd
import gzip
import pdb
from nltk import Tree
import os
from time import perf_counter
from typing import List
from stanfordcorenlp import StanfordCoreNLP
import math
import random
import pickle
import ast
import heapq
import sys
from tqdm import tqdm
from bisect import bisect_left

syntactic_ngram_path = "/projects/tir5/users/mengyan3/syntactic-ngrams/eng-1M" # read from /tmp
CORENLP_INSTALL_PATH = "/home/mengyan3/stanford-corenlp-4.4.0"
nlp = StanfordCoreNLP(CORENLP_INSTALL_PATH)

# idioms = set(pd.read_csv("./data/idioms/eng_all.csv")["idiom"])
idiom_df = pd.read_csv("./data/human_experiments/idioms_filtered.csv")
idioms = set(idiom_df.loc[(idiom_df["exclude"] != 1) & (idiom_df["pos_wrong"] != 1)]["text"])

#bigrams_1 = pd.read_csv("./data/ijcnlp_compositionality_data/freq.csv")
#bigrams_2 = pd.read_csv("./data/ramisch2016/compounds-lists/compounds-list-en.tsv", sep="\t")
#all_bigrams = set(list(bigrams_1["compound"])) | set(list(bigrams_2["compound"]))

all_bigrams_df = pd.read_csv("./data/human_experiments/bigrams_filtered.csv")
all_bigrams = set(all_bigrams_df.loc[(all_bigrams_df["exclude"] != 1) & (all_bigrams_df["pos_wrong"] != 1)]["text"])

CLAUSES = ["S", "SBAR", "SBARQ", "SINV", "SQ"]

def read_gzip_freqonly(gzip_dir: str, arc_type: str = "biarcs"):
    accumulated = 0
    ngrams = []
    texts = []
    freqs = []
    is_idiom = []
    for filename in os.listdir(gzip_dir):
        if not filename.endswith(".gz") or not arc_type in filename.split("."):
            continue
        gzip_path = os.path.join(gzip_dir, filename)
        with gzip.open(gzip_path, "rt") as f:
            for line in f:
                accumulated += 1
                split_line = line.split("\t")
                if accumulated % 10000000 == 0: # dump the lists every so often
                    print(accumulated)
                    with gzip.open("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed_idioms_new.gz", "at") as f:
                        f.write(f"syntactic_ngram\ttext\tfreq\tis_idiom\n")
                        for i, text in enumerate(texts):
                            f.write(f"{ngrams[i]}\t{text}\t{freqs[i]}\t{is_idiom[i]}\n")

                    # save just freqs separately?

                    ngrams = []
                    texts = []
                    freqs = []
                    is_idiom = []

                head_word, syntactic_ngram, total_count = split_line[0:3]
                split_ngram = syntactic_ngram.split()
                words_only = " ".join([w.split("/")[0] for w in split_ngram])
                ngrams.append(syntactic_ngram)
                texts.append(words_only)
                freqs.append(math.log(int(total_count)))
                is_idiom.append(words_only in idioms or words_only in all_bigrams)
    
    with gzip.open("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed_idioms_new.gz", "at") as f:
        f.write(f"syntactic_ngram\ttext\tfreq\tis_idiom\n")
        for i, text in enumerate(texts):
            f.write(f"{ngrams[i]}\t{text}\t{freqs[i]}\t{is_idiom[i]}\n")

def get_buckets(concated_gzip_path: str) -> List:
    big_lst = []
    idioms = []
    with gzip.open(concated_gzip_path, "rt") as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print(i)
            if i == 0:
                continue
            if i <= 550000000:
                continue
            if i == 550000000 * 2:
                break
            if "syntactic_ngram,text,freq,is_idiom\n" in line:
                continue
            #pdb.set_trace()
            split_line = line.split("\t")
            try:
                big_lst.append((i, float(split_line[2]), split_line[0]))
            except:
                print(split_line)
                continue
            if split_line[3] == "True\n":
                idioms.append((i, float(split_line[2]), split_line[0]))

    big_lst.sort(key=lambda x: x[1], reverse=True)
    with open("/projects/tir5/users/mengyan3/syntactic-ngrams/big_lst_1.p", "wb") as f:
        pickle.dump(big_lst, f)
    #pdb.set_trace()
    tenth = len(big_lst) // 10
    samples = []
    for i in range(0, 10):
        lst_slice = big_lst[i * tenth : (i + 1) * tenth]
        sample = random.sample(lst_slice, 1000)
        samples.extend(sample)
    samples.extend(idioms)
    return samples

def get_binparse_from_samples(samples: List, ngram_ind: int = 2):
    texts = []
    tree_types = []
    ngrams = []
    is_idiom = []
    freq = []
    texts_left = []
    texts_right = []
    for s in samples:
        ngram = s[ngram_ind]
        split_ngram = ngram.split()
        text = []
        pos_lst = []
        head_inds = []
        tree_branches = []
        skip_ngram = False

        if len(split_ngram) < 2:
            continue
        for ngram in split_ngram:
            ngram_parts = ngram.split("/")
            if len(ngram_parts) != 4:
                skip_ngram = True
                break
            word, pos, dep_tag, head_ind = ngram_parts[0], ngram_parts[1], ngram_parts[2], ngram_parts[3]
            text.append(word)
            pos_lst.append(pos)
            head_inds.append(int(head_ind))

        if (len(head_inds) != 0 and max(head_inds) >= 2) and not skip_ngram: # at least binary
            try:
                all_words = " ".join(text)
                print(all_words)
                if all_words in idioms:
                    is_idiom.append(True)
                else:
                    is_idiom.append(False)
                tree_str = nlp.parse(all_words)
                tree = Tree.fromstring(tree_str)
                tree.chomsky_normal_form()
                ind = 0
                tree_type = None
                l_text = None
                r_text = None

                if len(tree[0]) == 2:
                    parent_tree = tree[0]
                    tree_l = tree[0][0]
                    tree_r = tree[0][1]
                    l_text = " ".join(tree_l.leaves())
                    r_text = " ".join(tree_r.leaves())
                    tree_type = parent_tree.productions()[0]
                elif len(tree[0]) == 1:
                    parent_tree = tree[0][0]
                    tree_l = tree[0][0][0]
                    tree_r = tree[0][0][1]
                    l_text = " ".join(tree_l.leaves())
                    r_text = " ".join(tree_r.leaves())
                    tree_type = parent_tree.productions()[0]
        
                texts.append(all_words)
                ngrams.append(split_ngram)
                tree_types.append(tree_type)
                freq.append(s[1])
                texts_left.append(l_text)
                texts_right.append(r_text)
            except:
                continue

    df = pd.DataFrame({"text": texts, "ngram": ngrams, "left": texts_left, "right": texts_right, "log_freq": freq})
    df = df.sort_values("log_freq", ascending=False).drop_duplicates("text")
    df.to_csv("data/human_experiments/bigram_sample.csv", index=False)

    return df

def parse_text(all_words):
    tree_str = nlp.parse(all_words)
    tree = Tree.fromstring(tree_str)
    tree.chomsky_normal_form()
    l_text = None
    r_text = None

    if len(tree[0]) == 2:
        parent_tree = tree[0]
        tree_l = tree[0][0]
        tree_r = tree[0][1]
        l_text = " ".join(tree_l.leaves())
        r_text = " ".join(tree_r.leaves())
        tree_type = parent_tree.productions()[0]
    elif len(tree[0]) == 1:
        parent_tree = tree[0][0]
        tree_l = tree[0][0][0]
        tree_r = tree[0][0][1]
        l_text = " ".join(tree_l.leaves())
        r_text = " ".join(tree_r.leaves())

    return l_text, r_text

def get_idioms(concated_gzip_path: str) -> List:
    idiom_lst = []
    with gzip.open(concated_gzip_path, "rt") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i % 1000000 == 0:
                print(i)  
            split_line = line.split("\t")
            split_ngram = split_line[0].split() 
            words_only = " ".join([w.split("/")[0] for w in split_ngram])
            if words_only in idioms or words_only in all_bigrams:
                idiom_lst.append((i, float(split_line[2]), split_line[0]))
        
    return idiom_lst

def sample_similar_to_idioms(idiom_df: pd.DataFrame, concated_gzip_path: str, k: int = 50):
    ngrams = idiom_df["ngram"].tolist()
    matching_patterns = {}
    pos_freq = {}
    pos_dep_freq = {}
    for ngram in ngrams:
        ngram_lst = ast.literal_eval(ngram)
        pos_seq = [x.split("/")[1] for x in ngram_lst]
        pos_dep_seq = [" ".join(x.split("/")[1:3]) for x in ngram_lst]

        pos_str = " ".join(pos_seq)
        if pos_str not in pos_freq:
            pos_freq[pos_str] = 1
        else:
            pos_freq[pos_str] += 1

        pos_dep_str = ",".join(pos_dep_seq)
        if pos_dep_str not in pos_dep_freq:
            pos_dep_freq[pos_dep_str] = 1
        else:
            pos_dep_freq[pos_dep_str] += 1
        if pos_dep_str in matching_patterns:
            matching_patterns[pos_dep_str].append(ngram)
        else:
            matching_patterns[pos_dep_str] = [ngram]

    collected_phrases = {x: [] for x in ngrams}

    with gzip.open(concated_gzip_path, "rt") as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue
            if i % 1000000 == 0:
                print(i)  
            if i % 100000000 == 0:
                with open(f"./data/human_experiments/similar_sample_bigrams_{i}.p", "wb") as f:
                    pickle.dump(collected_phrases, f)
            split_line = line.split("\t")
            if len(split_line) != 4:
                continue
            ngram = split_line[0].split()
            ngram_key = str(ngram)
            try:
                pos_seq = [x.split("/")[1] for x in ngram]
                pos_dep_seq = [" ".join(x.split("/")[1:3]) for x in ngram]
                freq = float(split_line[2])
            except:
                continue

            pos_str = " ".join(pos_seq)
            pos_dep_str = ",".join(pos_dep_seq)

            if pos_dep_str not in matching_patterns:
                continue
            eligible_idioms = matching_patterns[pos_dep_str] 

            for idiom in eligible_idioms: 
                idiom_freq = float(idiom_df.loc[idiom_df["ngram"] == idiom]["log_freq"])
                freq_diff = abs(freq - idiom_freq)
                heap_val = (-freq_diff, ngram_key, freq)
                if len(collected_phrases[idiom]) < k or freq_diff < sys.float_info.epsilon:
                    heapq.heappush(collected_phrases[idiom], heap_val)
                else:
                    heapq.heappushpop(collected_phrases[idiom], heap_val)

    return collected_phrases

def sample_similar_to_idioms_fast(idiom_df: pd.DataFrame, concated_gzip_path: str, k: int = 50):
    idiom_ngrams = idiom_df["ngram"].tolist()
    freq_to_idiom = {}
    freqs_by_pos = {}
    for idiom in idiom_ngrams:
        ngram_lst = ast.literal_eval(idiom)
        pos_dep_str = ",".join([" ".join(x.split("/")[1:3]) for x in ngram_lst])
        
        idiom_freq = float(idiom_df.loc[idiom_df["ngram"] == idiom]["log_freq"])
        if pos_dep_str not in freq_to_idiom:
            freq_to_idiom[pos_dep_str] = {}

        if idiom_freq not in freq_to_idiom[pos_dep_str]:
            freq_to_idiom[pos_dep_str][idiom_freq] = [idiom]
        else:
            freq_to_idiom[pos_dep_str][idiom_freq].append(idiom)

        if pos_dep_str not in freqs_by_pos:
            freqs_by_pos[pos_dep_str] = [idiom_freq]
        else:
            freqs_by_pos[pos_dep_str].append(idiom_freq)
    for pos in freqs_by_pos:
        freqs_by_pos[pos] = sorted(set(freqs_by_pos[pos]))
    
    collected_phrases = {x: [] for x in idiom_ngrams}

    with gzip.open(concated_gzip_path, "rt") as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue
            if i % 100000000 == 0:
                with open(f"./data/human_experiments/similar_sample_bigrams_new_{i}.p", "wb") as f:
                    pickle.dump(collected_phrases, f)
            split_line = line.split("\t")
            if len(split_line) != 4:
                continue
            ngram = split_line[0].split()
            ngram_key = str(ngram)
            try:
                pos_dep_seq = [" ".join(x.split("/")[1:3]) for x in ngram]
                freq = float(split_line[2])
            except:
                continue

            pos_dep_str = ",".join(pos_dep_seq)
            if pos_dep_str not in freqs_by_pos:
                continue

            ind = bisect_left(freqs_by_pos[pos_dep_str], freq)

            for i in range(0, ind): # smaller freq
                idiom_freq = freqs_by_pos[pos_dep_str][i]
                # change this to handle multiple idioms with same freq
                idiom_key = freq_to_idiom[pos_dep_str][idiom_freq]
                freq_diff = abs(idiom_freq - freq)
                heap_val = (-freq_diff, ngram_key, freq)
                for curr_idiom_key in idiom_key:
                    if len(collected_phrases[curr_idiom_key]) < k or freq_diff < sys.float_info.epsilon:
                        heapq.heappush(collected_phrases[curr_idiom_key], heap_val)
                    else:
                        ret_val = heapq.heappushpop(collected_phrases[curr_idiom_key], heap_val)
                        if ret_val == heap_val:
                            break

            for i in range(ind, len(freqs_by_pos[pos_dep_str])): # larger freq
                idiom_freq = freqs_by_pos[pos_dep_str][i]
                # change this to handle multiple idioms with same freq
                idiom_key = freq_to_idiom[pos_dep_str][idiom_freq]
                freq_diff = abs(idiom_freq - freq)
                heap_val = (-freq_diff, ngram_key, freq)
                for curr_idiom_key in idiom_key:
                    if len(collected_phrases[curr_idiom_key]) < k or freq_diff < sys.float_info.epsilon:
                        heapq.heappush(collected_phrases[curr_idiom_key], heap_val)
                    else:
                        ret_val = heapq.heappushpop(collected_phrases[curr_idiom_key], heap_val)
                        if ret_val == heap_val:
                            break
    
    return collected_phrases
        

def read_gzip(gzip_dir: str, arc_type: str = "biarcs"):
    texts = []
    tree_types = []
    is_idiom = []
    freq = []
    texts_left = []
    texts_right = []
    for filename in os.listdir(gzip_dir):
        if not filename.endswith(".gz") or not arc_type in filename.split("."):
            continue
        gzip_path = os.path.join(gzip_dir, filename)
        j = 0
        with gzip.open(gzip_path, "rt") as f:
            for line in f:
                j += 1
                if j > 10000:
                    break
                line_split = line.split("\t")
                head_word, syntactic_ngram, total_count = line_split[0:3]
                split_ngram = syntactic_ngram.split()
                text = []
                pos_lst = []
                head_inds = []
                tree_branches = []
                skip_ngram = False

                if len(split_ngram) < 2:
                    continue
                for ngram in split_ngram:
                    ngram_parts = ngram.split("/")
                    if len(ngram_parts) != 4:
                        skip_ngram = True
                        break
                    word, pos, dep_tag, head_ind = ngram_parts[0], ngram_parts[1], ngram_parts[2], ngram_parts[3]
                    text.append(word)
                    pos_lst.append(pos)
                    head_inds.append(int(head_ind))
                
                if (len(head_inds) != 0 and max(head_inds) >= 2) and not skip_ngram: # at least binary
                    try:
                        all_words = " ".join(text)
                        if all_words in idioms:
                            is_idiom.append(True)
                        else:
                            is_idiom.append(False)
                        tree_str = nlp.parse(all_words)
                        tree = Tree.fromstring(tree_str)
                        tree.chomsky_normal_form()
                        ind = 0
                        tree_type = None
                        l_text = None
                        r_text = None

                        if len(tree[0]) == 2:
                            parent_tree = tree[0]
                            tree_l = tree[0][0]
                            tree_r = tree[0][1]
                            l_text = " ".join(tree_l.leaves())
                            r_text = " ".join(tree_r.leaves())
                            tree_type = parent_tree.productions()[0]
                        elif len(tree[0]) == 1:
                            parent_tree = tree[0][0]
                            tree_l = tree[0][0][0]
                            tree_r = tree[0][0][1]
                            l_text = " ".join(tree_l.leaves())
                            r_text = " ".join(tree_r.leaves())
                            tree_type = parent_tree.productions()[0]
                    # try:
                    #     finished = False
                    #     for subtree in tree:
                    #         if finished:
                    #             break
                    #         if subtree.label() in CLAUSES:
                    #             if subtree[0].label() in CLAUSES: # S -> S XP
                    #                 parent_tree = subtree
                    #                 l_tree = subtree[0][0]
                    #                 r_tree = subtree[1]
                    #             else: 
                    #                 if len(subtree) == 1: # S -> XP
                    #                     if subtree[0][0].label() == "S":
                    #                         continue
                    #                     if subtree[0][0].label() == "SINV":
                    #                         parent_tree = subtree[0][0]
                    #                         l_tree = subtree[0][0][0]
                    #                         r_tree = subtree[0][0][1]
                    #                     else:
                    #                         parent_tree = subtree[0]
                    #                         l_tree = subtree[0][0]
                    #                         r_tree = subtree[0][1]
                    #                 else: # S -> XP YP
                    #                     parent_tree = subtree
                    #                     l_tree = subtree[0]
                    #                     r_tree = subtree[1]
                    #             tree_type = parent_tree.productions()[0]
                    #             l_text = " ".join(l_tree.leaves())
                    #             r_text = " ".join(r_tree.leaves())
                    #             finished = True
                    # except:
                    #     print(f"skip: {text}")
                    #     continue
                    #     #pdb.set_trace()
                    
                        texts.append(all_words)
                        tree_types.append(tree_type)
                        freq.append(int(total_count))
                        #print(all_words)

                        
                        texts_left.append(l_text)
                        texts_right.append(r_text)
                    except: # parse failed
                        print("fail:", all_words)
                        continue

    df = pd.DataFrame({"text": texts, "left": texts_left, "right": texts_right, "tree_type": tree_types, "is_idiom": is_idiom, "freq": freq})
    df.to_csv(f"./data/human_experiments/{arc_type}.csv", index=False)

def bucket_by_logfreq():
    raise NotImplementedError

def sample_phrases(num_phrases: int):
    raise NotImplementedError

if __name__ == "__main__":
    idiom_gzip = "/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed_idioms_new.gz"
    bigram_gzip = "/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed_bigrams.gz"
    idiom_df = pd.read_csv("./data/human_experiments/idiom_sample.csv")
    bigram_df = pd.read_csv("./data/human_experiments/bigram_sample.csv")
    collected_idiom_sample = sample_similar_to_idioms_fast(bigram_df, bigram_gzip)
    with open("./data/human_experiments/similar_sample_bigrams_new.p", "wb") as f:
        pickle.dump(collected_idiom_sample, f)
    assert False
    #read_gzip_freqonly(syntactic_ngram_path, "arcs") 
    #assert False
    #samples = get_buckets("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed.gz")
    #with open("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/sampled-ngrams.p", "wb") as f:
        #pickle.dump(samples, f)
    samples = get_idioms("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M_fixed_bigrams.gz")
    with open("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/sampled-bigrams.p", "wb") as f:
        pickle.dump(samples, f)
    #assert False
    ngram_samples = pickle.load(open("/projects/tir5/users/mengyan3/syntactic-ngrams/freqs/sampled-bigrams.p", "rb"))
    get_binparse_from_samples(ngram_samples) 
    nlp.close()
