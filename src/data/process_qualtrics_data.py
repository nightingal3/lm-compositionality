import pandas as pd
import pdb
import torch
from scipy.stats import kendalltau, spearmanr
import numpy as np
import krippendorff as kd
from statsmodels.stats import inter_rater as irr
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import ast
from typing import List

from src.models.get_vecs import model_init, get_one_vec
from src.models.fit_composition_functions import AffineRegression

# phrases retroactively found to be wrong...
skip_phrases = ["dull as ditchwater", "ladies in-waiting"]

def get_human_answers(human_filepath: str, questions_filepath: str):
    df_qualtrics = pd.read_csv(human_filepath)
    df_questions = pd.read_csv(questions_filepath, names=["matched_idiom", "phrase", "left", "right"])
    question_cols = [col for col in df_qualtrics if "_Q" in col]
    overall_comp = {}
    l_r_comp = {}
    skip_next = 0
    for q_col in question_cols:
        if skip_next > 0:
            skip_next -= 1
            continue
        question_num = int(q_col.split("_")[0]) - 100
        question_text = df_questions.iloc[question_num]["phrase"]
        if "Q9" in q_col:
            if df_qualtrics[q_col].iloc[2] == "This phrase doesn't make sense or it isn't in English":
                skip_next = 2
                continue
            overall_comp[question_text] = int(df_qualtrics[q_col].iloc[2].split("-")[0]) # change this for multiple people
        elif "Q33" in q_col:
            question_subphrase = df_questions.iloc[question_num]["phrase"]
            if question_subphrase not in l_r_comp:
                l_r_comp[question_subphrase] = {}
            l_r_comp[question_subphrase]["l"] = int(df_qualtrics[q_col].iloc[2].split("-")[0]) # change this for multiple people
        elif "Q34" in q_col:
            if question_subphrase not in l_r_comp:
                l_r_comp[question_subphrase] = {}
            question_subphrase = df_questions.iloc[question_num]["phrase"]
            l_r_comp[question_subphrase]["r"] = int(df_qualtrics[q_col].iloc[2].split("-")[0]) # change this for multiple people


    return overall_comp, l_r_comp

def get_human_agreements(pilot_filepath: str, questions_filepath: str, find_best_subset: bool = False) -> float:
    df_qualtrics = pd.read_csv(pilot_filepath)
    df_qualtrics = df_qualtrics[(df_qualtrics["Finished"] == "True") & (df_qualtrics["Status"] != "Survey Preview")]
    df_questions = pd.read_csv(questions_filepath, names=["matched_idiom", "phrase", "left", "right"])
    question_cols = [col for col in df_qualtrics if "_Q9" in col]
    coders = np.zeros((len(df_qualtrics), len(question_cols)))
    phrase_order = []
    skip_next = [0] * len(df_qualtrics)
    
    l_r_comp = {}
    for j, q_col in enumerate(question_cols):
        question_num = int(q_col.split("_")[0]) - 100
        question_text = df_questions.iloc[question_num]["phrase"]
        phrase_order.append(question_text)
        for i, annot_resp in enumerate(df_qualtrics[q_col]):
            if annot_resp == "This phrase doesn't make sense or it isn't in English":
                coders[i][j] = np.nan
            else:
                coders[i][j] = int(df_qualtrics[q_col].iloc[i].split("-")[0])
    
    # drop all examples where it doesn't make sense to at least one person
    coders_df = pd.DataFrame(coders).dropna(axis=1, how="any")
    coders_no_nan = np.array(coders_df)

    overall_alpha = kd.alpha(reliability_data=coders_no_nan, level_of_measurement="ordinal")
    print("alpha: ", overall_alpha)

    if find_best_subset:
        all_combs = combinations(range(len(df_qualtrics)), 3)
        max_alpha = 0
        best_subset = None
        for comb in all_combs:
            subset_data = coders[comb, :]
            subset_alpha = kd.alpha(reliability_data=subset_data, level_of_measurement="ordinal")
            if subset_alpha > max_alpha:
                max_alpha = subset_alpha
                best_subset = comb
        print("overall: ", overall_alpha)
        print("best subset: ", best_subset, max_alpha)

    mean_scores = coders.mean(axis=0)

    phrase_to_score = {phrase: score for phrase, score in zip(phrase_order, mean_scores) if np.isfinite(score)}

    with open("mturk_avg_judgments.txt", "w") as f:
        for key, item in sorted(phrase_to_score.items(), key=lambda x: x[1]):
            f.write(f"{key},{item}\n")

    # get subphrase contributions
    subphrase_cols = [col for col in df_qualtrics if "_Q33" in col or "_Q34" in col]
    subphrases_l = [col for col in df_qualtrics if "Q33" in col]
    subphrases_r = [col for col in df_qualtrics if "Q34" in col]
    l_coders = np.zeros((len(df_qualtrics), len(subphrases_l)))
    r_coders = np.zeros((len(df_qualtrics), len(subphrases_r)))
    l_count, r_count = 0, 0

    for q_col in subphrase_cols:
        for i, annot_resp in enumerate(df_qualtrics[q_col]):
            question_num = int(q_col.split("_")[0]) - 100
            question_subphrase = df_questions.iloc[question_num]["phrase"]
            if question_subphrase not in l_r_comp:
                l_r_comp[question_subphrase] = {"l": None, "r": None}

            if "Q33" in q_col:
                try:
                    l_coders[i][l_count] = int(df_qualtrics[q_col].iloc[i].split("-")[0])
                except AttributeError:
                    l_coders[i][l_count] = np.nan
            elif "Q34" in q_col:
                try:
                    r_coders[i][r_count] = int(df_qualtrics[q_col].iloc[i].split("-")[0])
                except AttributeError:
                    r_coders[i][r_count] = np.nan

        if "Q33" in q_col:
            l_r_comp[question_subphrase]["l"] = l_coders[:, l_count].mean()
            l_count += 1
        elif "Q34" in q_col:
            l_r_comp[question_subphrase]["r"] = r_coders[:, r_count].mean()
            r_count += 1

    l_r_comp = {key: val for key, val in l_r_comp.items() if not np.isnan(val["l"]) and not np.isnan(val["r"])}

    return phrase_to_score, l_r_comp

def get_model_deviations(model_type: str, questions_filepath: str, emb_type: str = "CLS"):
    df_questions = pd.read_csv(questions_filepath, names=["matched_idiom", "phrase", "left", "right"])
    model, tokenizer = model_init(model_type, cuda=torch.cuda.is_available())
    if emb_type == "CLS": #these probably should have been the same dimension to start with...oh well
        sim = torch.nn.CosineSimilarity(dim=1)
    else:
        sim = torch.nn.CosineSimilarity(dim=0)
    composition_model = AffineRegression(input_size=2, output_size=768, cuda=torch.cuda.is_available())
    composition_model.load_state_dict(torch.load(f"./data/{model_type}_affine_{emb_type}_results/binary_trees/full/model_0_bintree.pt"))
    overall_comp = {}
    l_r_comp = {}

    for row in df_questions.to_dict(orient="records"):
        phrase, left_phrase, right_phrase = row["phrase"], row["left"], row["right"]
        if model_type == "gpt2":
            model_type_for_vec = "gpt"
        else:
            model_type_for_vec = "encoder"

        top_vec = get_one_vec(phrase, model, tokenizer, emb_type, cuda=torch.cuda.is_available(), model_type=model_type_for_vec)
        left_vec = get_one_vec(left_phrase, model, tokenizer, emb_type, cuda=torch.cuda.is_available(), model_type=model_type_for_vec)
        right_vec = get_one_vec(right_phrase, model, tokenizer, emb_type, cuda=torch.cuda.is_available(), model_type=model_type_for_vec)
        child_vecs = torch.stack((left_vec, right_vec)).squeeze(1)
        composed_vec = composition_model(child_vecs)
        if emb_type == "avg" and len(composed_vec.shape) > 1:
            composed_vec = composed_vec.squeeze(0)
        cos_sim = sim(top_vec, composed_vec)
        overall_comp[phrase] = cos_sim.item()
        if phrase not in l_r_comp:
                l_r_comp[phrase] = {}
        l_r_comp[phrase]["l"] = sim(top_vec, left_vec).item()
        l_r_comp[phrase]["r"] = sim(top_vec, right_vec).item()

    
    with open("model_comp_scores.txt", "w") as f:
        for key, item in sorted(overall_comp.items(), key=lambda x: x[1]):
            f.write(f"{key},{item}\n")

    return overall_comp, l_r_comp

def find_rank_correlation(human_answers: dict, model_answers: dict):
    order = sorted(human_answers.keys())
    human_sorted = [human_answers[key] for key in order]
    model_sorted = [model_answers[key] for key in order]
    corr, p_val = spearmanr(human_sorted, model_sorted)
    return corr, p_val

def find_other_correlations(human_answers, model_answers, idiom_datapath, questions_datapath):
    idiom_df = pd.read_csv(idiom_datapath)
    idiom_df["phrase_words"] = idiom_df["ngram"].apply(lambda x: " ".join([item.split("/")[0] for item in ast.literal_eval(x)]))
    idiom_df["idiom_words"] = idiom_df["assoc_idiom"].apply(lambda x: " ".join([item.split("/")[0] for item in ast.literal_eval(x)]))

    qs_df = pd.read_csv(questions_datapath)
    qs_df.columns = ["assoc_idiom", "phrase", "left", "right"]
    idiom_info_orig = pd.read_csv("data/human_experiments/idioms_filtered.csv")
    bigram_info_orig = pd.read_csv("data/human_experiments/bigrams_filtered.csv")
    noncomp_df = pd.concat([idiom_info_orig, bigram_info_orig])

    human_scores, model_scores = [], []
    freqs = []
    word_lens = []
    for phrase in human_answers:
        if phrase in skip_phrases:
            continue
        human_comp_score = human_answers[phrase]
        model_comp_score = model_answers[phrase]
        human_scores.append(human_comp_score)
        model_scores.append(model_comp_score)
        word_lens.append(len(phrase.split()))

        sel_row = idiom_df.loc[idiom_df["phrase_words"] == phrase]
        if len(sel_row) == 0:
            sel_row = noncomp_df.loc[noncomp_df["text"] == phrase]
        freq = sel_row["freq"] if "freq" in sel_row else sel_row["log_freq"]

        if len(freq) > 1: 
            try:
                if "freq" in sel_row:
                    freq = sel_row.loc[sel_row["pos_wrong"] != 1.0]["freq"]  
                else:
                    freq = sel_row.loc[sel_row["pos_wrong"] != 1.0]["log_freq"]
            except:
                try:
                    matched_idiom = qs_df.loc[qs_df["phrase"] == phrase].iloc[0]["assoc_idiom"]
                except:
                    pdb.set_trace()
                if "freq" in sel_row:
                    freq = sel_row.loc[sel_row["idiom_words"] == matched_idiom]["freq"]
                else:
                    freq = sel_row.loc[sel_row["idiom_words"] == matched_idiom]["log_freq"]
        try:
            freqs.append(freq.item())
        except:
            # should be the same freq in this case
            freqs.append(freq.iloc[0].item())
    print("human/model frequency corr")
    r_human_freq = spearmanr(human_scores, freqs)
    r_model_freq = spearmanr(model_scores, freqs)
    print(r_human_freq)
    print(r_model_freq)

    print("human/model len corr")
    r_human_len = spearmanr(human_scores, word_lens)
    r_model_len = spearmanr(model_scores, word_lens)
    
    human_corrs = {"length": r_human_len[1], "frequency": r_human_freq[1]}
    model_corrs = {"length": r_model_len[1], "frequency": r_model_freq[1]}
    return human_corrs, model_corrs


def subphrase_test(human_answers_l_r, model_answers_l_r):
    total_num = 0
    correct = 0
    left_closer = 0
    right_closer = 0
    correct_lst = []
    wrong_lst = []
    for phrase in human_answers_l_r:
        if human_answers_l_r[phrase]["l"] == human_answers_l_r[phrase]["r"]:
            continue
        if human_answers_l_r[phrase]["l"] > human_answers_l_r[phrase]["r"]:
            total_num += 1
            if model_answers_l_r[phrase]["l"] > model_answers_l_r[phrase]["r"]:
                left_closer += 1
            else:
                right_closer += 1
            is_correct = model_answers_l_r[phrase]["l"] > model_answers_l_r[phrase]["r"]
            correct += int(is_correct)
        if human_answers_l_r[phrase]["l"] < human_answers_l_r[phrase]["r"]:
            total_num += 1
            if model_answers_l_r[phrase]["l"] > model_answers_l_r[phrase]["r"]:
                left_closer += 1
            else:
                right_closer += 1
            is_correct = model_answers_l_r[phrase]["l"] < model_answers_l_r[phrase]["r"]
            correct += int(is_correct)
        
        if is_correct:
            correct_lst.append(phrase)
        else:
            wrong_lst.append(phrase)

    return correct/total_num, correct_lst, wrong_lst

def log_subphrase_test(questions_file: str, idiom_datapath: str, correct_model_phrases: List, wrong_model_phrases: List, human_ref_scores: dict, human_ref_l_r: dict, out_file: str) -> None:
    q_df = pd.read_csv(questions_file, names=["matched_idiom", "phrase", "left", "right"])
    idiom_df = pd.read_csv(idiom_datapath)
    idiom_df["phrase_words"] = idiom_df["ngram"].apply(lambda x: " ".join([item.split("/")[0] for item in ast.literal_eval(x)]))
    idiom_df["idiom_words"] = idiom_df["assoc_idiom"].apply(lambda x: " ".join([item.split("/")[0] for item in ast.literal_eval(x)]))

    with open(out_file, "w") as out_f:
        out_f.write("phrase,correct,is_idiom,human_comp_score,closer_meaning_human,closer_meaning_model\n")
        for p in correct_model_phrases:
            human_score = human_ref_scores[p]
            phrase_info = q_df.loc[q_df["phrase"] == p]
            #if p == "flannelled fool":
                #pdb.set_trace()
            try:
                is_idiom = (phrase_info["matched_idiom"] == phrase_info["phrase"]).item()
                closer = phrase_info["left"].item() if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info["right"].item()
                pdb.set_trace()
                freq = idiom_df.loc[idiom_df["phrase_words"] == p]["freq"]
            except: # selected as idiom and matched phrase
                is_idiom = (phrase_info.iloc[0]["matched_idiom"] == phrase_info.iloc[0]["phrase"])
                closer = phrase_info.iloc[0]["left"] if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info.iloc[0]["right"]
            model_closer = closer
            out_f.write(f"{p},correct,{is_idiom},{human_score},{closer},{model_closer}\n")
        for p in wrong_model_phrases:
            human_score = human_ref_scores[p]
            phrase_info = q_df.loc[q_df["phrase"] == p]
            try:
                is_idiom = (phrase_info["matched_idiom"] == phrase_info["phrase"]).item()
                closer = phrase_info["left"].item() if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info["right"].item()
                model_closer = phrase_info["right"].item() if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info["left"].item()

            except: # selected as idiom and matched phrase
                is_idiom = (phrase_info.iloc[0]["matched_idiom"] == phrase_info.iloc[0]["phrase"])
                closer = phrase_info.iloc[0]["left"] if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info.iloc[0]["right"]
                model_closer = phrase_info.iloc[0]["right"] if human_ref_l_r[p]["l"] > human_ref_l_r[p]["r"] else phrase_info.iloc[0]["left"]

            out_f.write(f"{p},wrong,{is_idiom},{human_score},{closer},{model_closer}\n")


def idiomaticity_test(questions_file: str, human_scores: dict, model_scores: dict):
    q_df = pd.read_csv(questions_file, names=["matched_idiom", "phrase", "left", "right"])
    example_count = 0
    correct = 0
    for idiom_phrase in q_df["matched_idiom"].unique():
        other_phrases = q_df.loc[q_df["matched_idiom"] == idiom_phrase]
        for other in other_phrases.to_dict(orient="records"):
            if other["phrase"] == idiom_phrase:
                continue
            if other["phrase"] not in human_scores or idiom_phrase not in human_scores:
                continue
            if human_scores[other["phrase"]] > human_scores[idiom_phrase]:
                example_count += 1
                if model_scores[other["phrase"]] > model_scores[idiom_phrase]:
                    correct += 1
    return correct/example_count

if __name__ == "__main__":
    human_file = "./data/qualtrics_results/idiom_annotation_final.csv"
    questions_file = "./data/qualtrics_results/final_qualtrics_1000_sample.csv"
    idioms_file = "./data/human_experiments/random_sample_filtered_3_match.csv"

    models = ["bert", "roberta", "deberta", "gpt2"]
    emb_types = ["CLS", "avg"]
    corr_human_len = None
    corr_human_freq = None
    corr_model_len = []
    corr_model_freq = []
    p_vals = []
    
    for model in models:
        for emb_type in emb_types:
            print("===")
            print(model, emb_type)
            print("===")
            overall_comp_human, l_r_comp_human = get_human_agreements(human_file, questions_file)
            overall_comp_model, l_r_comp_model = get_model_deviations(model, questions_file, emb_type=emb_type)
            subphrase_test_acc, correct_phrases, wrong_phrases = subphrase_test(l_r_comp_human, l_r_comp_model)
            print("subphrase acc: ", subphrase_test_acc)
            acc_idiom_test = idiomaticity_test(questions_file, overall_comp_human, overall_comp_model)
            print("idiomatic acc: ", acc_idiom_test)
            corr, p_val = find_rank_correlation(overall_comp_human, overall_comp_model)
            print("spearman correlation human/model: ", corr, p_val)
            p_vals.append(p_val)
            corr_human, corr_model = find_other_correlations(overall_comp_human, overall_comp_model, idioms_file, questions_file)
            if corr_human_len is None:
                corr_human_len = corr_human["length"]
            if corr_human_freq is None:
                corr_human_freq = corr_human["frequency"]
            corr_model_len.append(corr_model["length"])
            corr_model_freq.append(corr_model["frequency"])

    print(multipletests(p_vals, method="holm"))
    print("length")
    print(multipletests([corr_human_len] + corr_model_len, method="holm"))
    print("freq")
    print(multipletests([corr_human_freq] + corr_model_freq, method="holm"))