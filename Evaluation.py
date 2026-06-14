import numpy as np
from scipy.stats import spearmanr

import Threshold_Strategies as ts
from Background import get_chronological_max_indices

import time
import AutoBPMN_AI_Service as AutoBPMN


def text_similarity(sim_matrix, ground_truth):
    
    #calculates the Spearman Correlation between the clustered similarity scores (0-5)
    #and the ground-truth scores (from gen_GTs, 0-5) for each sentence-task pair

    pairs = []
    sim = []
    gt = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            pairs.append(f"S{i + 1}-T{j + 1}")
            sim.append(sim_matrix[i, j])
            gt.append(ground_truth[i, j])

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
        #AutoBPMN.process(body)

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
        #AutoBPMN.process(body)
            
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
    model_ids = ["01"]
   
    LEMMATIZE = False
    REMOVE_COND = False
    STRATEGY = 1 

    EMBEDDING_METHOD   = "bert"
    METRIC             = "cos"  
    TRADITIONAL_METHOD = None # only used if EMBEDDING_METHOD is None

    # Consensus parameters
    CONSENSUS_METHODS = [
        {"traditional": "levenshtein"},
        {"embedding": "bert", "metric": "cos"}
    ]

    RUN_TEXT_SIM  = False
    RUN_MODEL2TEXT = False
    RUN_BEST_OF_TUPLE = False
    RUN_TUPLE = False
    RUN_CONSENSUS = False
    RUN_BENCHMARK = True
  
    if not RUN_BENCHMARK:
        
        # build method config
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

            # ______ 4. Tuple Matching ______
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