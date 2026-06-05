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
    #for each column i (task), mark only the best matching row (sentence) as predicted TP
    rows, cols = sim_matrix.shape
    predicted = np.zeros((rows, cols), dtype=bool)
    best_indices = get_chronological_max_indices(sim_matrix, threshold)
    for i, best_i in enumerate(best_indices):
        if best_i != -1:
            predicted[best_i, i] = True

    gt_positive = (ground_truth == 1)

    # Jaccard Indx
    intersection=np.sum(predicted&gt_positive) #bitwis & operator due to matrix 
    union=np.sum(predicted|gt_positive)
    jaccard_index=intersection/union if union>0 else 0.0
    
    # F1
    tp=np.sum(predicted&gt_positive)
    fp=np.sum(predicted & (gt_positive == False))
    fn=np.sum((predicted == False) & gt_positive)
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=(2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    return round(jaccard_index,2), round(f1,2)


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
            "bpmn_xml": bpmn_xml,
            "approach": "model2text",
            "methods": [cfg],
        }
        
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
                    "methods": [{"traditional": "jaccard"},{"embedding": "gemini", "metric": "cos"}],
                }
            label = "JACCARD + GEMINI+COS"
        else:
            body = {
                "similarity_panel": True,
                "text": text,
                "bpmn_xml": bpmn_xml,
                "approach": approach,
                "methods": [{"embedding": "gemini", "metric": "cos"}],
            }
            label = "GEMINI+COS"
            
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
        [0, 0, 0, 2, 1, 5, 1, 0],
    ]),
    # Model 02
    "02": np.array([
        [5, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 5, 5, 2, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 5, 2, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 5, 2, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 5, 4, 5, 4, 4, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 5],
    ]),
    # Model 03
    "03": np.array([
        [5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 5, 5, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 5, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 5, 5, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 2, 4, 5, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 5, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 5, 4],
    ]),
    # Model 04
    "04": np.array([
        [5, 5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 5, 2, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 5, 5, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 5, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 5, 2, 0, 1],
        [1, 1, 0, 0, 1, 0, 3, 5, 3, 0],
        [0, 0, 0, 0, 0, 1, 1, 2, 4, 5],
    ]),
    # Model 05
    "05": np.array([
        [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 5, 5, 4, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 5, 5, 5, 2, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 5, 4],
    ]),
    # Model 06
    "06": np.array([
        [5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 5, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 5, 5, 5, 3, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 5, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 5, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 5, 2, 3, 5],
    ]),
    # Model 07
    "07": np.array([
        [5, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        [2, 5, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 5, 5, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 4, 1, 2, 0, 0, 0],
        [1, 1, 0, 0, 1, 5, 5, 5, 5, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 5],
    ]),
    # Model 08
    "08": np.array([
        [5, 1, 1, 0, 1, 0, 1, 0, 0],
        [2, 5, 4, 2, 2, 0, 0, 0, 0],
        [0, 0, 1, 5, 5, 1, 5, 2, 1],
        [0, 0, 0, 0, 2, 4, 3, 2, 0],
        [0, 0, 0, 0, 0, 2, 3, 5, 2],
        [0, 0, 1, 0, 0, 0, 0, 3, 5],
    ]),
    # Model 09
    "09": np.array([
        [4, 2, 0, 1, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 5, 2, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 5, 5, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 5, 4, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 5, 5],
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 5, 5, 1],  
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 5, 5],
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
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 2, 0],
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
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 5],
    ]),
    # Model 15
    "15": np.array([
        [5, 2, 0, 0, 0, 0, 0, 0],
        [3, 5, 5, 1, 2, 0, 0, 0],
        [0, 0, 3, 5, 2, 0, 0, 0],
        [0, 0, 1, 2, 5, 5, 1, 1],
        [0, 1, 0, 0, 0, 1, 5, 5],
    ]),
    # Model 16
    "16": np.array([
        [5, 1, 1, 0],
        [2, 4, 2, 0],
        [0, 3, 4, 5],
    ]),
    # Model 17
    "17": np.array([
        [5, 1, 1, 0],
        [2, 5, 2, 2],
        [0, 2, 5, 5],
    ]),
    # Model 18
    "18": np.array([
        [5, 1, 1],
        [1, 5, 2],
        [2, 2, 5],
    ]),
    # Model 19
    "19": np.array([
        [5, 2, 1, 0, 1],  
        [1, 0, 5, 0, 0],  
        [0, 5, 0, 0, 0],  
        [0, 0, 0, 5, 0],  
        [0, 0, 0, 0, 5],  
    ]),
    # Model 20
    "20": np.array([
        [5, 5, 1, 0, 0, 0],  
        [0, 0, 5, 0, 0, 1],  
        [0, 0, 1, 0, 0, 5],  
        [0, 0, 0, 5, 5, 0], 
    ]),
    # Model 21
    "21": np.array([
        [5, 4, 0, 0, 0, 0, 0],  
        [0, 0, 5, 1, 0, 1, 0],  
        [0, 0, 0, 5, 0, 0, 0],  
        [0, 0, 0, 0, 5, 5, 0],  
        [0, 0, 0, 0, 0, 0, 5], 
    ]),
    # Model 102 Perfect Correlation Test
    "102": np.array([
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 5],
    ]),
}

def get_gen_GT(doc_id):

    return gen_GT.get(doc_id)


if __name__ == "__main__":
    import Datasets

    # --- CONFIGURATION ---
    model_ids = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21"]#"01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21"
   
    LEMMATIZE = False
    REMOVE_COND = False
    STRATEGY = 1  # 1: F1-Gap, 2: Diagonal GT, 3: Generated GT

    # Choose embedding method ("bert", "gemini", "llm2vec") or set to None for traditional
    EMBEDDING_METHOD   = "bert"
    METRIC             = "cos"   # only used for embedding methods
    TRADITIONAL_METHOD = None # only used if EMBEDDING_METHOD is None
    # ---------------------
    """
    for doc_id in model_ids:
        if EMBEDDING_METHOD is not None:
            method_label = f"{EMBEDDING_METHOD.upper()} + {METRIC.upper()}"
            current_cfg  = {"embedding": EMBEDDING_METHOD, "metric": METRIC}
        else:
            method_label = TRADITIONAL_METHOD.upper()
            current_cfg  = {"traditional": TRADITIONAL_METHOD}

        print(f"\nEvaluating Model {doc_id} with method '{method_label}'...")

        # Load data and compute similarity matrix
        data_dict  = ts.load_data(current_cfg, [doc_id], LEMMATIZE, REMOVE_COND)
        sim_matrix = ts.get_sim_matrix(data_dict[doc_id], current_cfg)

        # Binary GT (0/1) for model2text evaluation
        gt_binary = Datasets.get_ground_truth(doc_id)

        # 3. Find optimal threshold using the chosen strategy
        if STRATEGY == 1:
            best_t, _ = ts.strategy1(current_cfg, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
        elif STRATEGY == 2:
            best_t, _ = ts.strategy2(current_cfg, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
        elif STRATEGY == 3:
            best_t, _ = ts.strategy3(current_cfg, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)

        print("Opt thres (Strat " + str(STRATEGY) + " over train): " + str(round(best_t,2)))

        # text sim
        cor,p=text_similarity(sim_matrix,get_gen_GT(doc_id))
        print(" SpearmanCorrel: "+str(cor)+ ", p-val:"+str(p))

        jaccard,f1=model2text_similarity(sim_matrix,gt_binary,best_t)

        print(f"______Model2Text Similarity Evaluation___")
        print(f"Jaccard Index: {jaccard}")
        print(f"F1 Score:      {f1}")
    """
    
   #____________________Benchmark
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

    benchmark_runtime(TEXT, BPMN_XML)