import numpy as np
from scipy.stats import spearmanr

import Threshold_Strategies as ts
from Background import get_chronological_max_indices

import time
import AutoBPMN_AI_Service as AutoBPMN


def text_similarity(sim_matrix, ground_truth, printing=True):
    #calculates the Spearman Correlation
    pairs = []
    sim = []
    gt = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            pairs.append(f"S{i + 1}-T{j + 1}")
            sim.append(sim_matrix[i, j])
            gt.append(ground_truth[i, j])

    if printing:
        # Combine into a list of tuples for sorting and printing
        data_points = list(zip(pairs, gt, sim))
        # Sort by human rating (gt) descending
        data_points.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'Sentence-task pair':<15} | {'GT-rating (0-5)':<27} | {'Computed-Similarity':<17}")
        print("-------------------------------------------------------------------------")
               
        for pair_name, gt_val, sim_val in data_points:
            print(f"{pair_name:<18} | {gt_val:<27} | {sim_val:<17}")
            
    sim=np.array(sim)
    gt=np.array(gt)
    
    correlation, p_value =spearmanr(sim,gt)
    return correlation,p_value

def print_spearman_table(doc_ids):
    ALL_METHODS = [
        {"traditional": "levenshtein"},
        {"traditional": "jaccard"},
        {"traditional": "wordnet"},
        {"traditional": "tfidf"},
        {"traditional": "word2vec"},
        {"embedding": "bert", "metric": "cos"},
        {"embedding": "bert", "metric": "eu"},
        {"embedding": "bert", "metric": "man"},
        {"embedding": "gemini", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "eu"},
        {"embedding": "llm2vec", "metric": "man"},
    ]

    combinations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    ]

    print(f"\n{'Spearman Correlation Table for Docs: ' + ', '.join(doc_ids)}")
    print("-" * 125)
    print(f"{'Method':<25} | {'None (Lem=F, RCE=F)':<18} | {'Lem (Lem=T, RCE=F)':<18} | {'RCE (Lem=F, RCE=T)':<18} | {'Lem+RCE (Lem=T, RCE=T)':<18} | {'Row Average':<18}")
    print("-" * 125)

    col_sums = [0] * len(combinations)
    col_counts = [0] * len(combinations)

    rows = []
    for method in ALL_METHODS:
        label = get_label(method)
        
        row_vals = []
        for col_idx, (lemmatize, remove_cond) in enumerate(combinations):
            avg_corr = 0
            docs = 0
            
            for doc_id in doc_ids:
                data_dict = ts.load_data(method, [doc_id], lemmatize, remove_cond)
                data = data_dict[doc_id]
                sim_matrix = ts.get_sim_matrix(data, method)
                ground_truth = get_gen_GT(doc_id)
                
                correlation, _ = text_similarity(sim_matrix, ground_truth, printing=False)
                
                avg_corr += correlation
                docs += 1
            
            avg_corr /= docs
            row_vals.append(avg_corr)
            
            col_sums[col_idx] += avg_corr
            col_counts[col_idx] += 1
                
        row_avg = sum(row_vals) / len(row_vals)
        rows.append((label, row_vals, row_avg))

    
    rows.sort(key=lambda x: x[2], reverse=True)

    for label, row_vals, row_avg in rows:
        row_str = f"{label:<25}"
        for val in row_vals:
            row_str += f" | {val:<18.2f}"
        row_str += f" | {row_avg:<18.2f}"
        print(row_str)
        
    print("-" * 125)
    
    col_str = f"{'Column Average':<25}"
    total_sum = 0
    total_count = 0
    for s, c in zip(col_sums, col_counts):
        c_avg = s / c if c > 0 else 0
        col_str += f" | {c_avg:<18.2f}"
        total_sum += c_avg
        total_count += 1
        
    overall_avg = total_sum / total_count 
    col_str += f" | {overall_avg:<18.2f}"
    print(col_str)
    print("-" * 125)

def print_model2text_table(doc_ids, strategy=1):
    import Datasets
    ALL_METHODS = [
        {"traditional": "levenshtein"},
        {"traditional": "jaccard"},
        {"traditional": "wordnet"},
        {"traditional": "tfidf"},
        {"traditional": "word2vec"},
        {"embedding": "bert", "metric": "cos"},
        {"embedding": "bert", "metric": "eu"},
        {"embedding": "bert", "metric": "man"},
        {"embedding": "gemini", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "eu"},
        {"embedding": "llm2vec", "metric": "man"},
    ]

    combinations = [
        (False, False),
        (True, False),
        (False, True),
        (True, True)
    ]

    print(f"\n{'Model2Text Jaccard / F1 Table for Docs: ' + ', '.join(doc_ids)}")
    print("-" * 145)
    print(f"{'Method':<25} | {'None (Lem=F, RCE=F)':<20} | {'Lem (Lem=T, RCE=F)':<20} | {'RCE (Lem=F, RCE=T)':<20} | {'Lem+RCE (Lem=T, RCE=T)':<20} | {'Row Average':<20}")
    print("-" * 145)

    col_sums_jac = [0] * len(combinations)
    col_sums_f1 = [0] * len(combinations)
    col_counts = [0] * len(combinations)

    rows = []
    for method in ALL_METHODS:
        label = get_label(method)
        
        row_vals = []  #list of (jac, f1) per combination
        for col_idx, (lemmatize, remove_cond) in enumerate(combinations):
            avg_jac = 0
            avg_f1 = 0
            docs = 0
            
            for doc_id in doc_ids:
                data_dict = ts.load_data(method, [doc_id], lemmatize, remove_cond)
                data = data_dict[doc_id]
                sim_matrix = ts.get_sim_matrix(data, method)
                ground_truth = Datasets.get_ground_truth(doc_id)
                best_t = ts.get_precomputed_threshold(method, strategy, lemmatize, remove_cond)
                
                jaccard, f1 = model2text_similarity(sim_matrix, ground_truth, best_t)
                
                avg_jac += jaccard
                avg_f1 += f1
                docs += 1
            
            avg_jac /= docs
            avg_f1 /= docs
            row_vals.append((avg_jac, avg_f1))
            
            col_sums_jac[col_idx] += avg_jac
            col_sums_f1[col_idx] += avg_f1
            col_counts[col_idx] += 1
                
        row_sum_jac = 0
        row_sum_f1 = 0
        for v in row_vals:
            row_sum_jac += v[0]
            row_sum_f1 += v[1]
        row_avg_jac = row_sum_jac / len(row_vals)
        row_avg_f1 = row_sum_f1 / len(row_vals)
        rows.append((label, row_vals, row_avg_jac, row_avg_f1))

   
    rows.sort(key=lambda x: x[3], reverse=True)#sort according to f1

    for label, row_vals, row_avg_jac, row_avg_f1 in rows:
        row_str = f"{label:<25}"
        for jac, f1 in row_vals:
            val_str = f"{jac:.2f} / {f1:.2f}"
            row_str += f" | {val_str:<20}"
        avg_str = f"{row_avg_jac:.2f} / {row_avg_f1:.2f}"
        row_str += f" | {avg_str:<20}"
        print(row_str)
        
    print("-" * 145)
    
    col_str = f"{'Column Average':<25}"
    total_sum_jac = 0
    total_sum_f1 = 0
    total_count = 0
    for s_jac, s_f1, c in zip(col_sums_jac, col_sums_f1, col_counts):
        c_avg_jac = s_jac / c if c > 0 else 0
        c_avg_f1 = s_f1 / c if c > 0 else 0
        c_str = f"{c_avg_jac:.2f} / {c_avg_f1:.2f}"
        col_str += f" | {c_str:<20}"
        total_sum_jac += c_avg_jac
        total_sum_f1 += c_avg_f1
        total_count += 1
        
    overall_avg_jac = total_sum_jac / total_count 
    overall_avg_f1 = total_sum_f1 / total_count 
    o_str = f"{overall_avg_jac:.2f} / {overall_avg_f1:.2f}"
    col_str += f" | {o_str:<20}"
    print(col_str)
    print("-" * 145)

def model2text_similarity(sim_matrix, ground_truth, threshold):
    rows, cols = sim_matrix.shape
    gt_positive = (ground_truth == 1)

    # for each column i (task), mark only the best matching row (sentence) as predicted TP
    predicted = np.zeros((rows, cols), dtype=bool)
    best_indices = get_chronological_max_indices(sim_matrix, threshold)
    for i, best_i in enumerate(best_indices):
        if best_i != -1:
            predicted[best_i, i] = True

    # reduce GT to one match per column, since model2text can only produce one match per column
    adjusted_gt = gt_positive.copy()
    for col in range(cols):
        gt_rows = np.where(gt_positive[:, col])[0]
        if len(gt_rows) > 1:
            # if prediction hits one of the GT matches, keep that one, otherwise keep the first
            if best_indices[col] != -1 and best_indices[col] in gt_rows:
                keep = best_indices[col]
            else:
                keep = gt_rows[0]
            adjusted_gt[:, col] = False
            adjusted_gt[keep, col] = True

    # Jaccard Index
    intersection=np.sum(predicted&adjusted_gt)
    union=np.sum(predicted|adjusted_gt)
    jaccard_index=intersection/union if union>0 else 0.0
    
    # F1
    tp=np.sum(predicted&adjusted_gt)
    fp=np.sum(predicted & (adjusted_gt == False))
    fn=np.sum((predicted == False) & adjusted_gt)
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=(2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    return round(jaccard_index,2),round(f1,2)


def best_of_tuple_eval(sim_matrix, groups, ground_truth, threshold):
    num_sentences = ground_truth.shape[0]
    num_tasks = ground_truth.shape[1]

    best_indices = get_chronological_max_indices(sim_matrix, threshold)

    # dissolve tuple matches (tasks) back into individual sentence-task pairs
    dissolved = np.zeros((num_sentences, num_tasks), dtype=float)
    for group_col, best_row in enumerate(best_indices):
        if best_row != -1:
            for task_index in groups[group_col]:
                if task_index < num_tasks:
                    dissolved[best_row, task_index] = 1.0
    return model2text_similarity(dissolved, ground_truth, threshold=0.5)


def tuple_eval(sim_matrix, sentence_ranges, task_ranges, ground_truth, threshold):
    num_sentences = ground_truth.shape[0]
    num_tasks = ground_truth.shape[1]

    best_indices = get_chronological_max_indices(sim_matrix, threshold)

    # dissolve tuple matches (sentences and tasks) back into individual sentence-task pairs
    dissolved = np.zeros((num_sentences, num_tasks), dtype=float)
    for task_tuple_col, best_sent_tuple_row in enumerate(best_indices):
        if best_sent_tuple_row != -1:
            s_start, s_end = sentence_ranges[best_sent_tuple_row]
            t_start, t_end = task_ranges[task_tuple_col]
            for si in range(s_start, s_end + 1):
                for ti in range(t_start, t_end + 1):
                    if si < num_sentences and ti < num_tasks:
                        dissolved[si, ti] = 1.0
    return model2text_similarity(dissolved, ground_truth, threshold=0.5)


def consensus_eval(consensus_sim, ground_truth, threshold):
    return model2text_similarity(consensus_sim, ground_truth, threshold)


def get_label(cfg):
    if "embedding" in cfg:
        return f"{cfg['embedding'].upper()}+{cfg['metric'].upper()}"
    return cfg["traditional"].upper()





def benchmark_runtime(text, bpmn_xml):
    LEMMATIZE = False
    REMOVE_COND = False

    ALL_METHODS = [
        {"traditional": "levenshtein"},
        {"traditional": "jaccard"},
        {"traditional": "wordnet"},
        {"traditional": "tfidf"},
        {"traditional": "word2vec"},
        {"embedding": "bert", "metric": "cos"},
        {"embedding": "bert", "metric": "eu"},
        {"embedding": "bert", "metric": "man"},
        {"embedding": "gemini", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "eu"},
        {"embedding": "llm2vec", "metric": "man"},
    ]

    APPROACHES = ["tuple", "best_of_tuple", "consensus"]

    

    #______text similiarity comparison___________
    print("_" * 70)
 
    RUNS=3
    m2t_results = []
    for cfg in ALL_METHODS:
        label = get_label(cfg)
        
        body = {
            "similarity_panel": True,
            "text": text,
            "bpmn_xml":bpmn_xml,
            "approach": "model2text",
            "methods":[cfg],
        }
        
        #preload models
        AutoBPMN.process(body)

        total_time = 0
        for _ in range(RUNS):
            start = time.perf_counter()
            AutoBPMN.process(body)
            total_time += (time.perf_counter() - start)
           
            
        avg_end = total_time / RUNS
        m2t_results.append((label, avg_end))
        
    print(f"{'Method':<25} {'Runtime (s)':>14}")
    print("_" * 70)
    m2t_results.sort(key=lambda x: x[1])
    for label, end in m2t_results:
        t_str = f"{end:.4f}"
        print(f"{label:<25} {t_str:>14}")
    
    print("_" * 70)

    fd_results = []
    for approach in APPROACHES:
        if approach == "consensus":
            body = {
                    "similarity_panel": True,
                    "text": text,
                    "bpmn_xml": bpmn_xml,
                    "approach": "consensus",
                    "methods": [{"traditional": "jaccard"},{"traditional": "levenshtein"}],
                }
            label = "JACCARD + LEVENSHTEIN"
        else:
            body = {
                "similarity_panel": True,
                "text": text,
                "bpmn_xml": bpmn_xml,
                "approach": approach,
                "methods": [{"traditional": "jaccard"}],
            }
            label = "JACCARD"
            
       #preload models
        AutoBPMN.process(body)
            
        total_time = 0
        for _ in range(RUNS):
            start = time.perf_counter()
            AutoBPMN.process(body)
            total_time += (time.perf_counter() - start)
            
        avg_end = total_time / RUNS
        fd_results.append((label, approach, avg_end))
           
    print(f"{'Method':<25} {'Approach':<20} {'Runtime (s)':>14}")
    print("_" * 70)
    fd_results.sort(key=lambda x: x[2])
    for label, approach, end in fd_results:
        t_str = f"{end:.4f}" 
        print(f"{label:<25} {approach:<20} {t_str:>14}")


gen_GT = {
    # Model 01
    "01": np.array([
        [5, 3, 1, 0, 0, 0, 0, 0],
        [2, 5, 5, 3, 1, 0, 0, 0],
        [0, 0, 2, 5, 2, 0, 0, 0],
        [0, 0, 0, 2, 5, 4, 2, 0],
        [0, 1, 1, 0, 0, 1, 5, 5],
        [0, 0, 0, 2, 1, 5, 1, 0]
    ]),
    # Model 02
    "02": np.array([
        [5, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 5, 5, 2, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 5, 2, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 5, 2, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 5, 4, 5, 4, 4, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 5]
    ]),
    # Model 03
    "03": np.array([
        [5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 5, 5, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 5, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 5, 5, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 4, 5, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 5, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 5, 4]
    ]),
    # Model 04
    "04": np.array([
        [5, 5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 5, 2, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 5, 5, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 5, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 5, 2, 0, 1],
        [1, 1, 0, 0, 1, 0, 3, 5, 3, 0],
        [0, 0, 0, 0, 0, 1, 1, 2, 4, 5]
    ]),
    # Model 05
    "05": np.array([
        [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 5, 5, 4, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 5, 5, 5, 2, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 5, 4]
    ]),
    # Model 06
    "06": np.array([
        [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 5, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 5, 5, 5, 3, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 5, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 5, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 5, 2, 3, 5]
    ]),
    # Model 07
    "07": np.array([
        [5, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 5, 5, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 4, 1, 2, 0, 0, 0],
        [1, 1, 0, 0, 1, 5, 5, 5, 5, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 5]
    ]),
    # Model 08
    "08": np.array([
        [5, 1, 1, 0, 1, 0, 1, 0, 0],
        [2, 5, 4, 2, 2, 0, 0, 0, 0],
        [0, 0, 1, 5, 5, 1, 5, 2, 1],
        [0, 0, 0, 0, 2, 4, 3, 2, 0],
        [0, 0, 0, 0, 0, 2, 3, 5, 2],
        [0, 0, 1, 0, 0, 0, 0, 3, 5]
    ]),
    # Model 09
    "09": np.array([
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 5, 2, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 5, 5, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 5, 4, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 5, 5]
    ]),
    # Model 10
    "10": np.array([
        [5, 5, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 1, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 5, 5, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 3, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 5, 3, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 5, 4, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 5, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 5, 5, 1]
    ]),
    # Model 11
    "11": np.array([
        [5, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 5, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 5, 5, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 5, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 2, 5, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 5, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 5, 2, 2, 5, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 3, 5, 1, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 5, 2, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 5, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 2, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 5, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 5, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 5, 5]
    ]),
    # Model 12
    "12": np.array([
        [5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 5, 5, 3, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 3, 5, 2, 1, 0, 0, 0],
        [0, 0, 1, 2, 5, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 2, 5, 5, 5, 5, 5],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]),
    # Model 13
    "13": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 4, 4, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 5, 5, 4, 3, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 5, 4, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 3, 4, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 5, 5, 1, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 4],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 0]
    ]),
    # Model 14
    "14": np.array([
        [5, 1, 2, 0, 0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 1],
        [2, 5, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 3, 5, 4, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 2, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 5, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 3, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 4, 5, 5, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 5]
    ]),
    # Model 15
    "15": np.array([
        [5, 2, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 1, 2, 0, 0, 0],
        [0, 0, 3, 5, 2, 0, 0, 0],
        [0, 0, 1, 2, 5, 5, 1, 1],
        [0, 1, 0, 0, 0, 1, 5, 5]
    ]),
    # Model 16
    "16": np.array([
        [5, 1, 1, 0],
        [2, 4, 2, 0],
        [0, 3, 4, 5]
    ]),
    # Model 17
    "17": np.array([
        [5, 1, 1, 0],
        [2, 5, 2, 2],
        [0, 2, 5, 5]
    ]),
    # Model 18
    "18": np.array([
        [5, 1, 1],
        [1, 5, 2],
        [2, 2, 5]
    ]),
    # Model 19
    "19": np.array([
        [5, 2, 1, 0, 1],  
        [1, 0, 5, 0, 0],  
        [0, 5, 0, 0, 0],  
        [0, 0, 0, 5, 0],  
        [0, 0, 0, 0, 5]
    ]),
    # Model 20
    "20": np.array([
        [5, 5, 1, 0, 0, 0],  
        [0, 0, 5, 0, 0, 1],  
        [0, 0, 1, 0, 0, 5],  
        [0, 0, 0, 5, 5, 0]
    ]),
    # Model 21
    "21": np.array([
        [5, 4, 0, 0, 0, 0, 0],  
        [0, 0, 5, 1, 0, 1, 0],  
        [0, 0, 0, 5, 0, 0, 0],  
        [0, 0, 0, 0, 5, 5, 0],  
        [0, 0, 0, 0, 0, 0, 5]
    ]),
    # Model 102 Perfect Correlation Test
    "102": np.array([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 5]
    ]),
}

def get_gen_GT(doc_id):

    return gen_GT.get(doc_id)


if __name__ == "__main__":
    import Datasets
    import Further_Dimension_Approaches as fda

    # --- CONFIGURATION ---
    model_ids = ["01","02","03","04","05","06","07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"]
   
    LEMMATIZE = False
    REMOVE_COND = False
    STRATEGY = 1 

    EMBEDDING_METHOD   = None #"bert"
    METRIC             = None#"cos"  
    TRADITIONAL_METHOD = "levenshtein" # only used if EMBEDDING_METHOD is None

    
    CONSENSUS_METHODS = [
        {"traditional": "levenshtein"},
        {"embedding": "bert", "metric": "cos"}
    ]

    RUN_TEXT_SIM  = False
    RUN_SPEARMAN_TABLE = True
    RUN_MODEL2TEXT = False
    RUN_MODEL2TEXT_TABLE = True
    RUN_BEST_OF_TUPLE = False
    RUN_TUPLE = False
    RUN_CONSENSUS = False
    RUN_BENCHMARK = False
  

    if RUN_SPEARMAN_TABLE:
        print_spearman_table(model_ids)
        
    if RUN_MODEL2TEXT_TABLE:
        print_model2text_table(model_ids, STRATEGY)
  
    if not RUN_BENCHMARK and not RUN_SPEARMAN_TABLE and not RUN_MODEL2TEXT_TABLE:
        if EMBEDDING_METHOD is not None:
            method_label = f"{EMBEDDING_METHOD.upper()} + {METRIC.upper()}"
            current_cfg  = {"embedding": EMBEDDING_METHOD, "metric": METRIC}
        else:
            method_label = TRADITIONAL_METHOD.upper()
            current_cfg  = {"traditional": TRADITIONAL_METHOD}

        for doc_id in model_ids:
            print(f"Evaluating Model {doc_id} with method '{method_label}'")

            data_dict  = ts.load_data(current_cfg, [doc_id], LEMMATIZE, REMOVE_COND)
            data       = data_dict[doc_id]
            sim_matrix = ts.get_sim_matrix(data, current_cfg)
            gt_binary  = Datasets.get_ground_truth(doc_id)
            best_t     = ts.get_precomputed_threshold(current_cfg, STRATEGY, LEMMATIZE, REMOVE_COND)

            # text Similarity _____
            if RUN_TEXT_SIM:
                cor, p = text_similarity(sim_matrix, get_gen_GT(doc_id))
                print(f"Text Similarity")
                print(f"  Spearman Correlation: {cor}")
                print(f"  p-value:              {p}")

            # _noraml Model2Text Similarity___
            if RUN_MODEL2TEXT:
                jaccard, f1 = model2text_similarity(sim_matrix, gt_binary, best_t)
                print(f"Model2Text Similarity")
                print(f"  Jaccard Index: {jaccard}")
                print(f"  GT-F1 Score:   {f1}")

            # Best-Of-Tuple Matching ___
            if RUN_BEST_OF_TUPLE:
                sim_best, groups = fda.best_of_tuple_matching(data, current_cfg)
                jaccard_bot, f1_bot = best_of_tuple_eval(sim_best, groups, gt_binary, best_t)
                print(f"Best-Of-Tuple Matching")
                print(f"  Jaccard Index: {jaccard_bot}")
                print(f"  GT-F1 Score:   {f1_bot}")

            # ______ Tuple Matching ______
            if RUN_TUPLE:
                sim_tuple, s_tuples, t_tuples, s_ranges, t_ranges = fda.tuple_matching(data, current_cfg)
                jaccard_tm, f1_tm = tuple_eval(sim_tuple, s_ranges, t_ranges, gt_binary, best_t)
                print(f"Tuple Matching")
                print(f"  Jaccard Index: {jaccard_tm}")
                print(f"  GT-F1 Score:   {f1_tm}")

            # Consensus Matching ______
            if RUN_CONSENSUS:
                consensus_sim, sentences, tasks, match_f1, method_labels = fda.consensus_matching(
                    doc_id, CONSENSUS_METHODS, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND
                )
                num_methods = len(CONSENSUS_METHODS)
                min_confidence = int(num_methods * 2 / 3)
                consensus_t = min_confidence / num_methods
                jaccard_cm, f1_cm = consensus_eval(consensus_sim, gt_binary, consensus_t)
                print(f"Consensus Matching")
                print(f"  Methods:       {', '.join(method_labels)}")
                print(f"  Jaccard Index: {jaccard_cm}")
                print(f"  GT-F1 Score:   {f1_cm}")
    # _____Benchmark 
    if RUN_BENCHMARK:
        TEXT = "The customer places an order. We receive the order and process the payment. Finally, the goods are shipped to the customer."
        BPMN_XML = """<testset xmlns="http://cpee.org/ns/properties/2.0">
  <description>
    <description xmlns="http://cpee.org/ns/description/1.0">
      <call id="a1" endpoint="auto">
        <parameters>
          <label>Receive customer order</label>
        </parameters>
      </call>
      <call id="a2" endpoint="auto">
        <parameters>
          <label>Check inventory</label>
        </parameters>
      </call>
      <call id="a3" endpoint="auto">
        <parameters>
          <label>Process payment</label>
        </parameters>
      </call>
      <call id="a4" endpoint="auto">
        <parameters>
          <label>Ship goods</label>
        </parameters>
      </call>
    </description>
  </description>
</testset>"""
        print(f"\n{'='*70}")
        print("Benchmark")
        print(f"{'='*70}")
        benchmark_runtime(TEXT, BPMN_XML)