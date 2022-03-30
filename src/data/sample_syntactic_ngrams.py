import pandas as pd
import gzip
import pdb
from nltk import Tree
import os
from time import perf_counter
from stanfordcorenlp import StanfordCoreNLP
import math
syntactic_ngram_path = "/projects/tir5/users/mengyan3/syntactic-ngrams/eng-1M" # read from /tmp
CORENLP_INSTALL_PATH = "/home/mengyan3/stanford-corenlp-4.4.0"
nlp = StanfordCoreNLP(CORENLP_INSTALL_PATH)

idioms = set(pd.read_csv("./data/idioms/eng.csv")["idiom"])
CLAUSES = ["S", "SBAR", "SBARQ", "SINV", "SQ"]

def read_gzip_freqonly(gzip_dir: str, arc_type: str = "biarcs"):
    ngrams = []
    texts = []
    freqs = []
    is_idiom = []
    i = 0
    for filename in os.listdir(gzip_dir):
        if not filename.endswith(".gz") or not arc_type in filename.split("."):
            continue
        gzip_path = os.path.join(gzip_dir, filename)
        with gzip.open(gzip_path, "rt") as f:
            for line in enumerate(f):
                i += 1
                line_split = line.split("\t")
                if i % 100000 == 0:
                    print(i)
                if i % 10000000 == 0: # dump the lists every so often
                    with gzip.open("./projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M.gz", "at") as f:
                        f.write(f"syntactic_ngram,text,freq,is_idiom")
                        for i, text in enumerate(texts):
                            f.write(f"{ngrams[i]},{text},{freqs[i]},{is_idiom[i]}\n")

                    # save just freqs separately
                    with open("")
                    
                    ngrams = []
                    texts = []
                    freqs = []
                    is_idiom = []

                head_word, syntactic_ngram, total_count = line_split[0:3]
                split_ngram = syntactic_ngram.split()
                words_only = " ".join([w.split("/")[0] for w in split_ngram])
                ngrams.append(syntactic_ngram)
                texts.append(words_only)
                freqs.append(math.log(int(total_count)))
                is_idiom.append(words_only in idioms)
    
    with gzip.open("./projects/tir5/users/mengyan3/syntactic-ngrams/freqs/freqs_eng_1M.gz", "at") as f:
        f.write(f"syntactic_ngram,text,freq,is_idiom")
        for i, text in enumerate(texts):
            f.write(f"{ngrams[i]},{text},{freqs[i]},{is_idiom[i]}\n")

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
    read_gzip_freqonly(syntactic_ngram_path, "biarcs")
    nlp.close()