import numpy as np
import pandas as pd
import seaborn as sns #heatmap dokumentation: https://seaborn.pydata.org/generated/seaborn.heatmap.html
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import Datasets


import New_And_State_of_the_art_Embeddings as embeddings
import Background as bg

from Background import (
    get_asymmetric_diagonal_row, get_chronological_max_indices,
    f1_score, compute_match_f1
)


#____________________ config _________________________________________________________________________________
TRAINING_IDS = ["01","02","03","04","05","06","07","08","09","10"] #--> training_IDs for calclating the optimal thresholds
VALIDATION_IDS = ["19","20","21"] #--> validation_IDs for testing the optimal thresholds


LEMMATIZE = False  
REMOVE_COND=False    

#either embedding or traditional

METHOD_CONFIG = {"embedding": "gemini", "metric": "cos"}
#METHOD_CONFIG = {"traditional": "levenshtein"}

#________________ what to show

# Show Score vs Threshold  curves
SHOW_CURVES= False

# Show validation heatmaps 
SHOW_HEATMAPS=  True

# Rank all methods
RANK_ALL=False

#________________ precomputed thresholds ____________________________

def method_key(method_config):
    if "embedding" in method_config:
        return method_config["embedding"] + "+" + method_config["metric"]
    return method_config["traditional"]

STRATEGY_1_THRESHOLDS = {
    ("levenshtein", False, False): 0.52,
    ("jaccard", False, False): 0.06,
    ("wordnet", False, False): 0.39,
    ("tfidf", False, False): 0.09,
    ("word2vec", False, False): 0.55,
    ("gemini+cos", False, False): 0.69,
    ("bert+cos", False, False): 0.48,
    ("bert+eu", False, False): 0.46,
    ("bert+man", False, False): 0.5,
    ("llm2vec+cos", False, False): 0.66,
    ("llm2vec+eu", False, False): 0.08,
    ("llm2vec+man", False, False): 0.28,
    ("levenshtein", False, True): 0.48,
    ("jaccard", False, True): 0.08,
    ("wordnet", False, True): 0.39,
    ("tfidf", False, True): 0.13,
    ("word2vec", False, True): 0.56,
    ("gemini+cos", False, True): 0.69,
    ("bert+cos", False, True): 0.46,
    ("bert+eu", False, True): 0.46,
    ("bert+man", False, True): 0.5,
    ("llm2vec+cos", False, True): 0.68,
    ("llm2vec+eu", False, True): 0.1,
    ("llm2vec+man", False, True): 0.31,
    ("levenshtein", True, False): 0.54,
    ("jaccard", True, False): 0.14,
    ("wordnet", True, False): 0.44,
    ("tfidf", True, False): 0.17,
    ("word2vec", True, False): 0.57,
    ("gemini+cos", True, False): 0.7,
    ("bert+cos", True, False): 0.48,
    ("bert+eu", True, False): 0.47,
    ("bert+man", True, False): 0.51,
    ("llm2vec+cos", True, False): 0.65,
    ("llm2vec+eu", True, False): 0.06,
    ("llm2vec+man", True, False): 0.28,
    ("levenshtein", True, True): 0.54,
    ("jaccard", True, True): 0.14,
    ("wordnet", True, True): 0.45,
    ("tfidf", True, True): 0.17,
    ("word2vec", True, True): 0.57,
    ("gemini+cos", True, True): 0.7,
    ("bert+cos", True, True): 0.48,
    ("bert+eu", True, True): 0.46,
    ("bert+man", True, True): 0.51,
    ("llm2vec+cos", True, True): 0.67,
    ("llm2vec+eu", True, True): 0.09,
    ("llm2vec+man", True, True): 0.29,
}
STRATEGY_2_THRESHOLDS = {
    ("levenshtein", False, False): 0.25,
    ("jaccard", False, False): 0.02,
    ("wordnet", False, False): 0.15,
    ("tfidf", False, False): 0.04,
    ("word2vec", False, False): 0.37,
    ("gemini+cos", False, False): 0.62,
    ("bert+cos", False, False): 0.31,
    ("bert+eu", False, False): 0.35,
    ("bert+man", False, False): 0.45,
    ("llm2vec+cos", False, False): 0.49,
    ("llm2vec+eu", False, False): 0.03,
    ("llm2vec+man", False, False): 0.22,
    ("levenshtein", False, True): 0.21,
    ("jaccard", False, True): 0.01,
    ("wordnet", False, True): 0.13,
    ("tfidf", False, True): 0.04,
    ("word2vec", False, True): 0.29,
    ("gemini+cos", False, True): 0.62,
    ("bert+cos", False, True): 0.33,
    ("bert+eu", False, True): 0.34,
    ("bert+man", False, True): 0.45,
    ("llm2vec+cos", False, True): 0.5,
    ("llm2vec+eu", False, True): 0.04,
    ("llm2vec+man", False, True): 0.24,
    ("levenshtein", True, False): 0.24,
    ("jaccard", True, False): 0.02,
    ("wordnet", True, False): 0.14,
    ("tfidf", True, False): 0.05,
    ("word2vec", True, False): 0.37,
    ("gemini+cos", True, False): 0.62,
    ("bert+cos", True, False): 0.33,
    ("bert+eu", True, False): 0.33,
    ("bert+man", True, False): 0.45,
    ("llm2vec+cos", True, False): 0.51,
    ("llm2vec+eu", True, False): 0.05,
    ("llm2vec+man", True, False): 0.25,
    ("levenshtein", True, True): 0.24,
    ("jaccard", True, True): 0.02,
    ("wordnet", True, True): 0.14,
    ("tfidf", True, True): 0.06,
    ("word2vec", True, True): 0.35,
    ("gemini+cos", True, True): 0.6,
    ("bert+cos", True, True): 0.32,
    ("bert+eu", True, True): 0.35,
    ("bert+man", True, True): 0.41,
    ("llm2vec+cos", True, True): 0.51,
    ("llm2vec+eu", True, True): 0.05,
    ("llm2vec+man", True, True): 0.25,
}
STRATEGY_3_THRESHOLDS = {
    ("levenshtein", False, False): 0.53,
    ("jaccard", False, False): 0.1,
    ("wordnet", False, False): 0.42,
    ("tfidf", False, False): 0.13,
    ("word2vec", False, False): 0.6,
    ("gemini+cos", False, False): 0.75,
    ("bert+cos", False, False): 0.59,
    ("bert+eu", False, False): 0.53,
    ("bert+man", False, False): 0.53,
    ("llm2vec+cos", False, False): 0.7,
    ("llm2vec+eu", False, False): 0.12,
    ("llm2vec+man", False, False): 0.32,
    ("levenshtein", False, True): 0.54,
    ("jaccard", False, True): 0.1,
    ("wordnet", False, True): 0.42,
    ("tfidf", False, True): 0.14,
    ("word2vec", False, True): 0.59,
    ("gemini+cos", False, True): 0.74,
    ("bert+cos", False, True): 0.57,
    ("bert+eu", False, True): 0.51,
    ("bert+man", False, True): 0.53,
    ("llm2vec+cos", False, True): 0.71,
    ("llm2vec+eu", False, True): 0.12,
    ("llm2vec+man", False, True): 0.33,
    ("levenshtein", True, False): 0.55,
    ("jaccard", True, False): 0.13,
    ("wordnet", True, False): 0.44,
    ("tfidf", True, False): 0.16,
    ("word2vec", True, False): 0.63,
    ("gemini+cos", True, False): 0.75,
    ("bert+cos", True, False): 0.59,
    ("bert+eu", True, False): 0.52,
    ("bert+man", True, False): 0.52,
    ("llm2vec+cos", True, False): 0.7,
    ("llm2vec+eu", True, False): 0.12,
    ("llm2vec+man", True, False): 0.32,
    ("levenshtein", True, True): 0.54,
    ("jaccard", True, True): 0.13,
    ("wordnet", True, True): 0.44,
    ("tfidf", True, True): 0.17,
    ("word2vec", True, True): 0.63,
    ("gemini+cos", True, True): 0.75,
    ("bert+cos", True, True): 0.59,
    ("bert+eu", True, True): 0.52,
    ("bert+man", True, True): 0.53,
    ("llm2vec+cos", True, True): 0.72,
    ("llm2vec+eu", True, True): 0.12,
    ("llm2vec+man", True, True): 0.32,
}

def get_precomputed_threshold(method_config, strategy=1, lemmatize=False, remove_cond=False):
    key = method_key(method_config)
    lookup_key = (key, lemmatize, remove_cond)
    #print("loading precomputed threshold")

    if strategy == 1:
        return STRATEGY_1_THRESHOLDS[lookup_key]
    elif strategy == 2:
        return STRATEGY_2_THRESHOLDS[lookup_key]
    elif strategy == 3:
        return STRATEGY_3_THRESHOLDS[lookup_key]


#_____________________________________________________________________________________________________________________________

def get_sim_matrix(data, method_config):
  
    if "embedding" in method_config:
        return embeddings.compute_similarity_matrix(
            method_config["metric"],
            data["emb_sentences"],
            data["emb_tasks"]
        )
    elif "traditional" in method_config:
        return bg.compute_similarity_matrix(
            method_config["traditional"],
            data["sentences"],
            data["tasks"]
        )


def load_data(method_config, ids, lemmatize, remove_cond, text=None, bpmn_xml=None):
    if "embedding" in method_config:
        m = method_config["embedding"]

        if m == "bert":
            return embeddings.get_bert_embeddings(ids, lemmatize, remove_cond, text=text, bpmn_xml=bpmn_xml)
        elif m == "gemini":
            return embeddings.get_gemini_embeddings(ids, lemmatize, remove_cond, text=text, bpmn_xml=bpmn_xml)
        elif m == "llm2vec":
            return embeddings.get_llm2vec_embeddings(ids, lemmatize, remove_cond, text=text, bpmn_xml=bpmn_xml)
    else:
        # Traditional methods
        res = {}
        for doc_id in ids:
            tasks, sentences = Datasets.get_data(doc_id=doc_id, text=text, bpmn_xml=bpmn_xml, lemmatized= lemmatize, remove_conditions=remove_cond)
            res[doc_id] ={"sentences": sentences, "tasks": tasks, "emb_sentences": None, "emb_tasks": None}
        return res


METRICS = []
training_data = {}
training_data_config = None  #tracks the configuration (e.g., {'embedding': 'bert', ...} or {'traditional': 'tfidf'}) last used when switching between different  methods                         
validation_data = {}


#_________________Strategy 1________________________
def get_mixed_sim_matrices(ids, method_config, data_source):

     #Compute all n x n similarity matrices for a list of ids,
    #mixing task embeddings from one ID with sentence embeddings from another

    sim_matrices = {}
    for i, id_task in enumerate(ids):
        for j, id_text in enumerate(ids):

            #cross mix sentence embeddings with task embeddings
            mixed_data = {
                "emb_sentences": data_source[id_text]["emb_sentences"],
                "emb_tasks": data_source[id_task]["emb_tasks"],
                "sentences": data_source[id_text]["sentences"],
                "tasks":data_source[id_task]["tasks"],
            }
            sim_matrices[(i, j)] = get_sim_matrix(mixed_data, method_config)
    return sim_matrices

def strategy1(method_config, step=0.01, lemmatize=False, remove_cond=False):
    #F1-Gap Maximization across different Models

    #if function invoked by other classes (traing data not loaded)
    global training_data, training_data_config
    if not training_data or training_data_config != method_config:
        print(f"Loading data with configuration: {method_config}")
        training_data = load_data(method_config,TRAINING_IDS, lemmatize, remove_cond)
        training_data_config =method_config

    n = len(TRAINING_IDS)

    # compute all n x n similarity matrices
    sim_matrices = get_mixed_sim_matrices(TRAINING_IDS, method_config, training_data)

    # grid search over thresholds
    thresholds= np.arange(0.01, 1.01, step)
    best_threshold =-1
    best_gap =-1

    for t in thresholds:
        #build n×n F1 matrix
        f1_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Match-based F1
                f1_matrix[i, j] = compute_match_f1(sim_matrices[(i, j)], t)


        #calculate average F1 score of diagonal and not diagnoal
        diag_f1_sum=0.0
        diag_f1_count=0
        off_diag_sum=0.0
        off_diag_count=0
        for i in range(n):
            for j in range(n):
                if i == j:
                    diag_f1_sum += f1_matrix[i, j]
                    diag_f1_count += 1
                else:  #incorrect pair
                    off_diag_sum+=f1_matrix[i,j]
                    off_diag_count+=1

        diag_f1=diag_f1_sum/diag_f1_count if diag_f1_count>0 else 0.0
        off_diag_f1=off_diag_sum/off_diag_count if off_diag_count>0 else 0.0

        gap = diag_f1 - off_diag_f1

        if gap > best_gap:
            best_gap = gap
            best_threshold = t

    return round(best_threshold, 2), round(best_gap, 2)



#_______________________Strategy 2_______________________________________________


# get_asymmetric_diagonal_row 
def f1_with_diagonal_GT(matrix, threshold, gt_variance):

    #Compute GT-based F1 using the asymmetric diagonal band
    #--> TP = cell value >= threshold AND cell is within the GT diagonal band
    
    rows,cols=matrix.shape

    tp=0
    fp=0
    fn=0

    for j in range(rows):
        for i in range(cols):
            val=matrix[j,i]

            #GT: distance from expected row is within gt_variance
            expected_row = get_asymmetric_diagonal_row(i, rows, cols)
            is_gt_match = (abs(j - expected_row) <= gt_variance)

            is_positive =(val >= threshold)

            if is_gt_match and is_positive:
                tp += 1
            elif is_positive and not is_gt_match:
                fp += 1
            elif is_gt_match and not is_positive:
                fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return f1_score(p, rec)


def strategy2(method_config, step=0.01, lemmatize=False, remove_cond=False):
    #GT-F1 Max with Diagonal approximation across different Models

    #if function invoked by other classes (traing data not loaded)
    global training_data, training_data_config
    if not training_data or training_data_config != method_config:
        print(f"Loading data with configuration: {method_config}")
        training_data = load_data(method_config, TRAINING_IDS, lemmatize, remove_cond)
        training_data_config = method_config

    optimal_thresholds = []
    best_f1_list = []

    for doc_id in TRAINING_IDS:

        data = training_data[doc_id]
        sim = get_sim_matrix(data, method_config)
        rows, cols = sim.shape[0], sim.shape[1]
        #variance: 20% of rows
        current_v = int(rows * 0.2)

        #grid search for best F1 threshold
        thresholds = np.arange(0.01, 1.01, step)
        best_t = -1
        best_f1 = -1

        for t in thresholds:
            f1 = f1_with_diagonal_GT(sim, t, current_v)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        optimal_thresholds.append(best_t)
        best_f1_list.append(best_f1)

        #print(f"Model {doc_id}: optimal threshold = {best_t:.2f}, max F1 = {best_f1:.2f}")

    #final threshold = average across all matrices
    avg_threshold = np.mean(optimal_thresholds)
    avg_f1 = np.mean(best_f1_list)


    return round(avg_threshold, 2), round(avg_f1, 2)


#___________________Strategy 3_______________________________

def f1_with_gen_GT(sim_matrix, ground_truth, threshold):
    
    #Compute GT-based F1 using binary GT
    #--> TP = cell value >= threshold AND ground_truth[j, i] == 1
    
    rows,cols=sim_matrix.shape

    tp=0
    fp=0
    fn=0

    for j in range(rows):
        for i in range(cols):
            val = sim_matrix[j, i]

            #GT
            is_gt_match = (ground_truth[j, i] == 1)

            #prediction: cell value >= threshold
            is_positive =(val >= threshold)

            if is_gt_match and is_positive:
                tp += 1
            elif is_positive and not is_gt_match:
                fp += 1
            elif is_gt_match and not is_positive:
                fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return f1_score(p, rec)


def strategy3(method_config, step=0.01, lemmatize=False, remove_cond=False):

    # GT-F1 Maximization with True gen GT
    #-->Like Strategy 2 but replaces the diagonal approximation
  
    #if function invoked by other classes (traing data not loaded)
    global training_data, training_data_config
    if not training_data or training_data_config != method_config:
        print(f"Loading data with configuration: {method_config}")
        training_data = load_data(method_config, TRAINING_IDS, lemmatize, remove_cond)
        training_data_config = method_config

    optimal_thresholds = []
    best_f1_list = []

    for doc_id in TRAINING_IDS:
        gt = Datasets.get_ground_truth(doc_id)

        data = training_data[doc_id]
        sim = get_sim_matrix(data, method_config)

        #Grid search for best F1 threshold
        thresholds = np.arange(0.01, 1.01, step)
        best_t = -1
        best_f1 = -1

        for t in thresholds:
            f1 = f1_with_gen_GT(sim, gt, t)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        optimal_thresholds.append(best_t)
        best_f1_list.append(best_f1)

        #print(f"Model {doc_id}: optimal threshold = {best_t:.2f}, max F1 = {best_f1:.2f}")
        

    # Final threshold = average across all matrices
    avg_threshold = np.mean(optimal_thresholds)
    avg_f1 = np.mean(best_f1_list)
    return round(avg_threshold, 2),round(avg_f1, 2)


#________________Heatmaps_______________________________

def draw_single_basic_heatmap(ax, similarity,sentences,tasks, threshold, title, n_models=1, custom=False):
    #basic heatmap (threshold + match boxes)
    # ax is passed here to draw this specific heatmap inside the side-by-side window layout
    
    #wrap width for sentence and task labels so they fit into window
    wrap_width_s = 50
    wrap_width_t = 30
    labels_s=[]
    for j, s in enumerate(sentences):
        labels_s.append(f"S{j + 1}: {textwrap.fill(s, width=wrap_width_s)}")
      #if labels already have T1 T2.. inside (see best-of-tuple)
    labels_t=[]
    if custom:
        for t in tasks: labels_t.append(textwrap.fill(t, width=wrap_width_t))
    else:
        for i, t in enumerate(tasks): labels_t.append(f"T{i + 1}: {textwrap.fill(t, width=wrap_width_t)}")
    df = pd.DataFrame(similarity, index=labels_s, columns=labels_t)

    #colormap and  values below threshold grey
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_under('lightgray')

    #dynamic font size for cells (depends on number of sentences and tasks)
    max_dim = max(df.shape[0], df.shape[1])
    if max_dim <= 5:      
        font_size = 25
    elif max_dim <= 10:   
        font_size = 20
    elif max_dim <= 20:   
        font_size = 15
    elif max_dim <= 30:   
        font_size = 10
    else:                 
        font_size = 8
    
    #scale font size if multiple models
    scale = 1.0 / max(1, n_models ** 0.8) if n_models > 1 else 1.0
    font_size = font_size * scale

    #draw heatmap
    sns.heatmap(df,cmap=cmap,annot=True,fmt=".2f",linewidths=.5,ax=ax,vmin=threshold-1e-5,vmax=1.0,annot_kws={"size":font_size})

    best_indices = get_chronological_max_indices(df.values, threshold)

    #black bounding boxes around the best match for each column
    for col_index in range(df.shape[1]):
        row_index = best_indices[col_index]
        if row_index != -1:
            ax.add_patch(plt.Rectangle((col_index, row_index), 1, 1, fill=False, edgecolor='black', lw=2))
    
    #title and sentence and task label font sizes
    ax.set_title(title, fontsize=max(14, 20 * scale), fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size*0.7)

    #rotate labels so they are not cutted off
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Dynamic font size for T and F1 (scales with plot size, but at least 8pt)
    info_fs = max(8, 14 * scale)
    #display threshold
    ax.text(0.01, 1.01, f"T={threshold:.2f}", transform=ax.transAxes, fontsize=info_fs, color='gray', verticalalignment='bottom')
    #display F1 score
    f1_val = compute_match_f1(similarity, threshold)
    ax.text(0.99, 1.01, f"F1={f1_val:.2f}", transform=ax.transAxes, fontsize=info_fs, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F2B5DF', edgecolor='gray', alpha=0.9))


def draw_single_strategy2_heatmap(ax, similarity, sentences, tasks, threshold, title, n_models=1):
    #Draw a Strategy 2 heatmap (diagonal GT band)
    #draw on the subplots axes side-by-side in one window
    
    #wrap width for sentence and task labels so they fit into window
    wrap_width_s = 50
    wrap_width_t = 30
    labels_s=[]
    for j,s in enumerate(sentences):
        labels_s.append(f"S{j+1}: {textwrap.fill(s,width=wrap_width_s)}")
    labels_t=[]
    for i,t in enumerate(tasks):
        labels_t.append(f"T{i+1}: {textwrap.fill(t,width=wrap_width_t)}")
    df = pd.DataFrame(similarity, index=labels_s, columns=labels_t)

    #colormap and  values below threshold grey
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_under('lightgray')

    #dynamic font size for cells (depends on number of sentences and tasks)
    num_rows, num_cols = similarity.shape
    max_dim = max(num_rows, num_cols)
    if max_dim <= 5:      
        font_size = 25
    elif max_dim <= 10:   
        font_size = 20
    elif max_dim <= 20:   
        font_size = 15
    elif max_dim <= 30:   
        font_size = 10
    else:                 
        font_size = 8
    
    #scale font size if multiple models
    scale = 1.0 / max(1, n_models ** 0.8) if n_models > 1 else 1.0
    font_size = font_size * scale

    #draw heatmap
    sns.heatmap(df,cmap=cmap,annot=True,fmt=".2f",linewidths=.5,ax=ax,vmin=threshold-1e-5,vmax=1.0,annot_kws={"size":font_size})

    # calculate the variance (thickness) of the diagonal ground-truth band (e.g., 20% of rows)
    gt_variance = max(0, int(num_rows * 0.2))

    #Green/Red bounding boxes for all cells >= threshold
    for col_index in range(num_cols):
        for row_index in range(num_rows):
            val = df.values[row_index, col_index]
            expected_row = get_asymmetric_diagonal_row(col_index, num_rows, num_cols)

            if val >= threshold - 1e-5:
                # TP (Green): Match is >= Threshold and inside the GT band (distance <= variance)
                # FP (Red): Match is >=Threshold but outside the GT band
                color = 'green' if abs(row_index - expected_row) <= gt_variance else 'red'
                ax.add_patch(plt.Rectangle((col_index, row_index), 1, 1, fill=False, edgecolor=color, lw=2))

    # FN (Blue, dashed)
    #boxes for all cells that belong to the GT band but were not detected as a match (>= Threshold)
    for i in range(num_cols):
        expected_row = get_asymmetric_diagonal_row(i, num_rows, num_cols)
        #iterate through range of rows that are considered GT matches for task i
        # max(0, ) and min(num_rows, ) ensure we stay within the matrix limits
        for j in range(max(0, expected_row - gt_variance), min(num_rows, expected_row + gt_variance + 1)):
            is_match = (df.values[j, i] >= threshold - 1e-5) #to ensure rounding consistency
            if not is_match:
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='blue', lw=1, linestyle='--'))

    #title and sentence and task label font sizes
    ax.set_title(title, fontsize=max(14, 20 * scale), fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size*0.7)

    #rotate labels so they are not cutted off
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Dynamic font size for T & F1 (scales with plot size, but at least 8pt)
    info_fs = max(8, 14 * scale)
    #display threshold
    ax.text(0.01, 1.01, f"T={threshold:.2f}", transform=ax.transAxes, fontsize=info_fs, color='gray', verticalalignment='bottom')
    #display F1 score
    f1_val = f1_with_diagonal_GT(similarity, threshold, gt_variance)
    ax.text(0.99, 1.01, f"F1={f1_val:.2f}", transform=ax.transAxes, fontsize=info_fs, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F2B5DF', edgecolor='gray', alpha=0.9))


def draw_single_strategy3_heatmap(ax, similarity, ground_truth, sentences, tasks, threshold, title, n_models=1):
    #Strategy 3 heatmap (gen GT)
    r = similarity.shape[0]
    c = similarity.shape[1]

    #wrap width for sentence and task labels so they fit into window
    wrap_width_s = 50
    wrap_width_t = 30
    labels_s=[]
    for j,s in enumerate(sentences[:r]):
        labels_s.append(f"S{j+1}: {textwrap.fill(s,width=wrap_width_s)}")
    labels_t=[]
    for i,t in enumerate(tasks[:c]):
        labels_t.append(f"T{i+1}: {textwrap.fill(t,width=wrap_width_t)}")
    df = pd.DataFrame(similarity,index=labels_s, columns=labels_t)

    #colormap and  values below threshold grey
    cmap =plt.colormaps["YlGnBu"].copy()
    cmap.set_under('lightgray')

    #dynamic font size for cells (depends on number of sentences and tasks)
    max_dim = max(r, c)
    if max_dim <= 5:
        font_size = 25
    elif max_dim <= 10:
        font_size = 20
    elif max_dim <= 20:
        font_size = 15
    elif max_dim <= 30:
        font_size = 10
    else:
        font_size = 8
    
    #scale font size if multiple models
    scale = 1.0 / max(1, n_models ** 0.8) if n_models > 1 else 1.0
    font_size = font_size * scale

    #draw heatmap
    sns.heatmap(df,cmap=cmap,annot=True,fmt=".2f",linewidths=.5,ax=ax,vmin=threshold-1e-5,vmax=1.0,annot_kws={"size":font_size})

    # Evaluate the matches against the GT matrix
    for col_index in range(c):
        for row_index in range(r):
            val = similarity[row_index, col_index]

            if val >= threshold - 1e-5:
                is_gt = (ground_truth[row_index, col_index] == 1)
                # TP (Green): Match is correct according to the GT
                # FP (Red): Match is incorrect according to the GT
                color = 'green' if is_gt else 'red'
                ax.add_patch(plt.Rectangle((col_index, row_index), 1, 1,fill=False, edgecolor=color, lw=2))

    # FN (Blue, dashed): Cells that should be a match (value = 1) according to the GT,
    # but were not detected by the model as >= Threshold
    for j in range(r):
        for i in range(c):
            if ground_truth[j, i] == 1:
                # Check if cell (j, i) was actually marked as a match
                if not (similarity[j, i] >= threshold - 1e-5):
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='blue', lw=1, linestyle='--'))

    #title and sentence and task label font sizes
    ax.set_title(title, fontsize=max(14, 20 * scale), fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size*0.7)

    #rotate labels so they are not cutted off
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Dynamic font size for T & F1 (scales with plot size, but at least 8pt)
    info_fs = max(8, 14 * scale)
    #display threshold
    ax.text(0.01, 1.01, f"T={threshold:.2f}", transform=ax.transAxes,
            fontsize=info_fs, color='gray', verticalalignment='bottom')
    #display F1 score
    f1_val = f1_with_gen_GT(similarity, ground_truth, threshold)
    ax.text(0.99, 1.01, f"F1={f1_val:.2f}",transform=ax.transAxes, fontsize=info_fs, verticalalignment='bottom', horizontalalignment='right',bbox=dict(boxstyle='round,pad=0.3', facecolor='#F2B5DF', edgecolor='gray', alpha=0.9))


#________________________Heatmaps in one window__________________________________________

#get the label for the method configuration
def get_label(method_config):
    if "embedding" in method_config:
        emb = method_config["embedding"].lower()
        metric = method_config["metric"].lower()
        if emb == "gemini":
            return "Gemini Embeddings"
        elif emb == "bert":
            return f"BERT Embeddings + {metric.upper()}"
        elif emb == "llm2vec":
            return f"LLM2Vec Embeddings + {metric.upper()}"
        else:
            return f"{method_config['embedding'].upper()} Embeddings + {metric.upper()}"
    return method_config.get("traditional", "unknown").upper()


def validate_all_strategies(method_config, t1, t2, t3):
    
    #4 windows
      #--> Window 1: Basic evaluation
      # --> Window 2: Strategy 1(F1Gap Maximization)
      # --> Window 3: Strategy 2(diagonal GT)
      # --> Window 4:Strategy 3(synthetic GT)
    
    n_val = len(VALIDATION_IDS)
    method_label = get_label(method_config)

    #_____ Window 1: Basic Evaluation ___
    # Shows normal heatmaps for all validation models using the calculated threshold t1
    
    # 1 row and n_val columns, which puts the model heatmaps side by side in one window
    fig1, axes1 = plt.subplots(1, n_val, figsize=(14.5 * n_val, 11), constrained_layout=True)
    # subplots gives array, but if there is only 1 model it returns one axis -->  wrap it in a list 
    if n_val == 1:
        axes1 = [axes1]
    fig1.suptitle(f"Basic Evaluation ({method_label}) - Threshold: {t1:.2f}", fontsize=16, fontweight='bold')

    for index, doc_id in enumerate(VALIDATION_IDS):
        data = validation_data[doc_id]
        sim = get_sim_matrix(data, method_config)
        #pass the specific subplot ax (axes1[index]) to the heatmap drawing function so it draws next to other heatmaps
        draw_single_basic_heatmap(axes1[index], sim, data["sentences"],data["tasks"],t1, f"Model {doc_id}",n_val)

    # _____ Window 2:Strategy 1 __
    sim_matrices_val = get_mixed_sim_matrices(VALIDATION_IDS, method_config, validation_data)

    f1_matrix_val = np.zeros((n_val, n_val))
    for i in range(n_val):
        for j in range(n_val):
            f1_matrix_val[i, j] = compute_match_f1(sim_matrices_val[(i, j)], t1)

    fig2,ax2 = plt.subplots(1, 1, figsize=(9, 7.2))
    fig2.suptitle(f"Strategy 1 Validation ({method_label}) - Threshold: {t1:.2f}",fontsize=16, fontweight='bold')
    labels_s1_y = []
    labels_s1_x = []
    for vid in VALIDATION_IDS:
        labels_s1_y.append(f"Model (Text) {vid}")
        labels_s1_x.append(f"Model (BPMN) {vid}")
    df_s1 = pd.DataFrame(f1_matrix_val, index=labels_s1_y, columns=labels_s1_x)
    cmap_s1 = plt.colormaps["YlGnBu"].copy()
    
    #dynmaic fontsize
    if n_val <= 5:
        font_size = 25
    elif n_val <= 10:
        font_size = 20
    elif n_val <= 20:
        font_size = 15
    elif n_val <= 30:
        font_size = 10
    else:
        font_size = 8
        
    sns.heatmap(df_s1, cmap=cmap_s1, annot=True, fmt=".2f", linewidths=.5,ax=ax2, annot_kws={"size": font_size})
    # Draw green boxes around the diagonal since these are the correct model-text pairs
    for i in range(n_val):
        ax2.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))
    ax2.set_title(f"Cross-Process F1 Matrix", fontsize=14)
    ax2.tick_params(axis='both', labelsize=font_size*0.7)
    plt.setp(ax2.get_yticklabels(), rotation=0)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])

    #____ Window 3: Strategy 2_________________-
    fig3, axes3 = plt.subplots(1, n_val, figsize=(14.5 * n_val, 11), constrained_layout=True)
    if n_val == 1:
        axes3 = [axes3]
    fig3.suptitle(f"Strategy 2 Validation ({method_label}) - Threshold: {t2:.2f}",fontsize=16, fontweight='bold')
    for id, doc_id in enumerate(VALIDATION_IDS):
        data = validation_data[doc_id]
        sim = get_sim_matrix(data, method_config)
        draw_single_strategy2_heatmap(axes3[id], sim, data["sentences"],data["tasks"], t2, f"Model {doc_id}", n_val)

    # legend for Strategy 2/3
    tp_patch = mpatches.Patch(edgecolor='green', fill=False, lw=2, label='TP')
    fp_patch = mpatches.Patch(edgecolor='red', fill=False, lw=2, label='FP')
    fn_patch = mpatches.Patch(edgecolor='blue', fill=False, lw=1, linestyle='--', label='FN')
    fig3.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower left', bbox_to_anchor=(0.0, 0.01), ncol=1,fontsize=12, frameon=False)

    #Window 4: Strategy 3 ____________________
    fig4, axes4 = plt.subplots(1, n_val, figsize=(14.5 * n_val, 11), constrained_layout=True)
    if n_val == 1:
        axes4 = [axes4]
    fig4.suptitle(f"Strategy 3 Validation ({method_label}) - Threshold: {t3:.2f}",fontsize=16, fontweight='bold')
    for index, doc_id in enumerate(VALIDATION_IDS):
        data = validation_data[doc_id]
        gt = Datasets.get_ground_truth(doc_id)
        sim = get_sim_matrix(data, method_config)
        draw_single_strategy3_heatmap(axes4[index], sim, gt, data["sentences"],data["tasks"], t3, f"Model {doc_id}", n_val)

    # legend for Strategy 3
    fig4.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower left', bbox_to_anchor=(0.0, 0.01), ncol=1,fontsize=12, frameon=False)
    plt.show()

#________________________Threshold curves___________________________________________

def plot_threshold_curves(method_config, t1, gap1, t2, f2, t3, f3, step=0.01):
    #visualize the threshold optimization (Score vs Threshold) for all 3 strategies in one window
    
    thresholds = np.arange(0.01, 1.01, step)
    n = len(TRAINING_IDS)
    method_label = get_label(method_config)

   
    #________- Strategy 1 Curve Data: F1-Gap Maximization
    sim_matrices = get_mixed_sim_matrices(TRAINING_IDS, method_config, training_data)

    strat1_scores = []
    # Evaluate every threshold to find the one that maximizes the gap
    for t in thresholds:
        f1_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                f1_matrix[i, j] = compute_match_f1(sim_matrices[(i, j)], t)

        # Calculate average F1 score of diagonal and off-diagonal
        diag_f1_sum =0.0
        diag_f1_count =0
        off_diag_sum =0.0
        off_diag_count =0
        for i in range(n):
            for j in range(n):
                if i == j:
                    diag_f1_sum += f1_matrix[i, j]
                    diag_f1_count +=1
                else:
                    off_diag_sum += f1_matrix[i, j]
                    off_diag_count += 1

        diag_f1 = diag_f1_sum / diag_f1_count if diag_f1_count > 0 else 0.0
        off_diag_f1 = off_diag_sum / off_diag_count if off_diag_count > 0 else 0.0

        
        gap = diag_f1 - off_diag_f1
        strat1_scores.append(gap)

    #_______- Strategy 2___
    strat2_doc_scores = {}
    for doc_id in TRAINING_IDS:
        strat2_doc_scores[doc_id] = []
        data = training_data[doc_id]
        sim = get_sim_matrix(data, method_config)
        rows, cols = sim.shape[0], sim.shape[1]
        # dynamic variance based on model size
        current_v = int(rows * 0.2)
        for t in thresholds:
            f1 = f1_with_diagonal_GT(sim, t, current_v)
            strat2_doc_scores[doc_id].append(float(f1))

    # calculate the average curve across all documents 
    strat2_avg_scores = np.mean(np.array(list(strat2_doc_scores.values())), axis=0)

    #_______-Strategy 3____
    strat3_doc_scores = {}
    for doc_id in TRAINING_IDS:
        gt = Datasets.get_ground_truth(doc_id)
        data = training_data[doc_id]
        sim = get_sim_matrix(data, method_config)
        strat3_doc_scores[doc_id] = []
        for t in thresholds:
            f1 = f1_with_gen_GT(sim, gt, t)
            strat3_doc_scores[doc_id].append(float(f1))

    #Average curve for gen-available documents
    strat3_avg_scores = np.mean(np.array(list(strat3_doc_scores.values())), axis=0)

    # _________plotting settings- 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Threshold Optimization Curves ({method_label})", fontsize=16, fontweight='bold')

    #plot 1
    axes[0].plot(thresholds, strat1_scores, color='blue', lw=2, label=f'F1-Gap (Max: {gap1:.2f})')
    #mark the max point
    axes[0].scatter(t1, gap1, color='blue', s=40, edgecolors='black', zorder=5)
    axes[0].text(t1, gap1 + 0.01, f"{gap1:.2f}", ha='center', va='bottom', fontsize=8, color='blue')

    # highlight the threshold
    axes[0].axvline(t1, color='red', linestyle='--', label=f'Calculated Threshold: {t1:.2f}')
    axes[0].set_title("Strategy 1: F1-Gap")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1-Gap")
    axes[0].legend()
    axes[0].grid(True)

    #____ plot 2____
    #  individual curves in light color and mark their maximums
    for doc_id, scores in strat2_doc_scores.items():
        max_index = np.argmax(scores)
        max_t = thresholds[max_index]
        max_f1 = scores[max_index]
        line, = axes[1].plot(thresholds, scores, alpha=0.3, label=f'Model {doc_id} (Max: {max_f1:.2f})')
        axes[1].scatter(max_t, max_f1, color=line.get_color(), s=40, edgecolors='black', zorder=5)
        axes[1].text(max_t, max_f1 + 0.01, f"{max_f1:.2f}", ha='center', va='bottom', fontsize=8, color=line.get_color())
    # plot the average curve in blue
    axes[1].plot(thresholds, strat2_avg_scores, color='blue', lw=2, label='Average Curve')
    axes[1].axvline(t2, color='red', linestyle='--', label=f'Calculated Threshold: {t2:.2f}')
    axes[1].set_title("Strategy 2: Diagonal GT F1")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True)

    #____ plot 3_________

    for doc_id, scores in strat3_doc_scores.items():
        max_index = np.argmax(scores)
        max_t = thresholds[max_index]
        max_f1= scores[max_index]
        line, = axes[2].plot(thresholds, scores, alpha=0.3, label=f'Model {doc_id} (Max: {max_f1:.2f})')
        axes[2].scatter(max_t, max_f1, color=line.get_color(), s=40, edgecolors='black', zorder=5)
        axes[2].text(max_t, max_f1 + 0.01, f"{max_f1:.2f}", ha='center', va='bottom', fontsize=8, color=line.get_color())

    axes[2].plot(thresholds, strat3_avg_scores, color='blue', lw=2, label='Average Curve')
    axes[2].axvline(t3, color='red', linestyle='--', label=f'Calculated Threshold: {t3:.2f}')
    axes[2].set_title("Strategy 3: gen GT F1")
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.show()


#___________rank methods according to best_f1_gap________________

def rank_all_methods(methods_to_rank, lemmatize=False, remove_cond=False):

    print("_" * 70)
    print(f"Method Ranking: (lemmatize= {lemmatize}, RCE= {remove_cond})")
    print("_" * 70)
    
    global training_data, training_data_config
    # Store original training data to restore it after ranking
    original_training_data = training_data.copy()
    original_config = training_data_config
    
    results = []
    for cfg in methods_to_rank:
        label = get_label(cfg)
        
        #switch training data to the current method's embeddings/text
        training_data = load_data(cfg, TRAINING_IDS, lemmatize, remove_cond)
        training_data_config = cfg
        
        t, gap = strategy1(cfg, lemmatize=lemmatize, remove_cond=remove_cond)
        results.append((label, gap, t))
      
    
    # restore original training data
    training_data = original_training_data
    training_data_config = original_config
    
    #Sort by gap descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"{'Method':<40} | {'Best Gap':<10} | {'Threshold':<10}")
    print("-" * 70)
    for label, gap, t in results:
        print(f"{label:<40} | {gap:<10.2f} | {t:<10.2f}")
    print("_" * 70)


def print_all_thresholds():
    ALL_METHODS = [
        {"traditional": "levenshtein"},
        {"traditional": "jaccard"},
        {"traditional": "wordnet"},
        {"traditional": "tfidf"},
        {"traditional": "word2vec"},
        {"embedding": "gemini", "metric": "cos"},
        {"embedding": "bert", "metric": "cos"},
        {"embedding": "bert", "metric": "eu"},
        {"embedding": "bert", "metric": "man"},
        {"embedding": "llm2vec", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "eu"},
        {"embedding": "llm2vec", "metric": "man"},
    ]

    global training_data, training_data_config

    for strategy_nr in [1, 2, 3]:
        strategy_dict = {}

        for lem in [False, True]:
            for rc in [False, True]:
                for cfg in ALL_METHODS:
                    key = method_key(cfg)

                    training_data = load_data(cfg, TRAINING_IDS, lem, rc)
                    training_data_config = cfg

                    if strategy_nr == 1:
                        t, _ = strategy1(cfg, lemmatize=lem, remove_cond=rc)
                    elif strategy_nr == 2:
                        t, _ = strategy2(cfg, lemmatize=lem, remove_cond=rc)
                    elif strategy_nr == 3:
                        t, _ = strategy3(cfg, lemmatize=lem, remove_cond=rc)

                    strategy_dict[(key, lem, rc)] = t

        #dictionary print
        print(f"STRATEGY_{strategy_nr}_THRESHOLDS = {{")
        for (key, lem, rc), t in strategy_dict.items():
            print(f'    ("{key}", {lem}, {rc}): {t},')
        print("}")
        print("_____________________________________________________")


if __name__ == "__main__":
    # Load initial data for the chosen configuration
    
    print(f"Loading initial data with configuration: {METHOD_CONFIG}")
    training_data = load_data(METHOD_CONFIG, TRAINING_IDS, LEMMATIZE, REMOVE_COND)
    validation_data = load_data(METHOD_CONFIG, VALIDATION_IDS, LEMMATIZE, REMOVE_COND)

    print("_" * 70)
    print("Threshold Strategy Results")
    print("_" * 70)

    t1, t2, t3 = None, None, None

    # _______Compute thresholds using the chosen configuration ___
    print(f"___ Method Config: {METHOD_CONFIG} ___")
    t1, gap1 = strategy1(METHOD_CONFIG, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    print(f"  Strategy 1 (F1-Gap Maximization):  threshold = {t1}  (max_gap = {gap1})")

    t2, f2 = strategy2(METHOD_CONFIG, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    print(f"  Strategy 2 (GT-F1 Max w/ Diagonal GT): threshold = {t2}  (max_f1 = {f2})")
    t3, f3 = strategy3(METHOD_CONFIG, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    print(f"  Strategy 3 (GT-F1 Max w/ gen GT): threshold = {t3}  (max_f1 = {f3})")

    # ___ method ranking ___
    
    ALL_METHODS = [
        {"traditional": "levenshtein"},
        {"traditional": "jaccard"},
        {"traditional": "wordnet"},
        {"traditional": "tfidf"},
        {"traditional":"word2vec"},
        {"embedding": "gemini", "metric": "cos"},
        {"embedding": "bert", "metric": "cos"},
        {"embedding": "bert", "metric": "eu"},
        {"embedding": "bert", "metric": "man"},
        {"embedding": "llm2vec", "metric": "cos"},
        {"embedding": "llm2vec", "metric": "eu"},
        {"embedding": "llm2vec", "metric": "man"}
    ]
    
    if RANK_ALL:
        rank_all_methods(ALL_METHODS, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)

    #_________plots
    
    if SHOW_CURVES:
        plot_threshold_curves(METHOD_CONFIG, t1, gap1, t2, f2, t3, f3)

    if SHOW_HEATMAPS:
        validate_all_strategies(METHOD_CONFIG, t1, t2, t3)

    
    #print_all_thresholds()



