import numpy as np
import Threshold_Strategies as ts
from Background import compute_match_f1, get_chronological_max_indices

import New_And_State_of_the_art_Embeddings as emb
import Background as bg

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap


# ____________________ config _________________________________________________________________________________
DOC_ID = "01"
STRATEGY = 1  # 1: F1-Gap, 2: Diagonal GT, 3: Generated GT
LEMMATIZE = False
REMOVE_COND = False

# either embedding or traditional
#METHOD_CONFIG = {"embedding": "gemini", "metric": "cos"}
METHOD_CONFIG = {"traditional": "levenshtein"}

# Consensus Parameters: list of method configs (embedding and/or traditional)
CONSENSUS_METHODS = [
    # {"embedding":"bert", "metric":"cos"},
    {"traditional": "levenshtein"},
    {"embedding": "gemini", "metric": "cos"}
    
]

# select which methods to run
RUN_CONSENSUS = False
RUN_TUPLE = False
RUN_BEST_OF_TUPLE = True

TEXT = None
"The customer places an order. We receive the order and process the payment. Finally, the goods are shipped to the customer."
    
BPMN_XML =  None
"""<testset xmlns="http://cpee.org/ns/properties/2.0">
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

# _____________________________________________________________________________________________________________________________


# ____________Consensus Matching_________________

def get_method_label(method_config):
    # returns a readable label for a method configuration
    if "embedding" in method_config:
        return f"{method_config['embedding'].upper()}+{method_config['metric'].upper()}"
    return method_config.get("traditional", "unknown").upper()




def consensus_matching(doc_id, method_configs, strategy=1, lemmatize=False, remove_cond=False, text=None, bpmn_xml=None):
    num_methods = len(method_configs)
    min_confidence = int(num_methods * 2 / 3)
    method_labels = []
    for cfg in method_configs:
        method_labels.append(get_method_label(cfg))


    # ______Match Extraction______
    # for each method:load data, compute similarity, get its threshold, extract thresholded matches
    all_best_indices = []
    method_thresholds = []
    sentences, tasks = None, None

    for cfg in method_configs:
        data = ts.load_data(cfg, [doc_id], lemmatize, remove_cond, text=text, bpmn_xml=bpmn_xml)[doc_id]
        if sentences is None:
            sentences,tasks=data["sentences"],data["tasks"]

        sim = ts.get_sim_matrix(data, cfg)

        # get the optimal threshold for this individual method
        if "embedding" in cfg:
            t = emb.get_threshold(cfg["embedding"], cfg["metric"], strategy, lemmatize, remove_cond)
        else:
            t = bg.get_threshold(cfg["traditional"], strategy, lemmatize, remove_cond)
        method_thresholds.append(t)
        #print(f"  {get_method_label(cfg)}: threshold = {t:.2f} (Strategy {strategy})")

        best_indices=get_chronological_max_indices(sim,t)
        all_best_indices.append(best_indices)

    num_rows = len(sentences)
    num_tasks = len(tasks)

    # convert to 2D array: shape (num_methods, num_tasks)
    # each row contains the proposed sentence index for each task by one method(-1 = no match)
    all_best_indices = np.array(all_best_indices)

    # ______Consensus Similarity Matrix (Jaccard Index)____________
    # for each cell (sentence, task), count how many methods proposed this match
    vote_matrix = np.zeros((num_rows, num_tasks))
    for method_index in range(num_methods):
        for j in range(num_tasks):
            matched_row = all_best_indices[method_index, j]
            if matched_row != -1:  # only count if above the method's threshold
                vote_matrix[matched_row, j] += 1

    # build the consensus similarity matrix using the Jaccard Index
    # each cell = (number of methods confirming this match) / (total number of methods)
    consensus_sim=vote_matrix/num_methods

    # zero out cells that do not meet the minimum confirmation threshold
    confidence_condition = vote_matrix >= min_confidence
    consensus_sim_filtered = np.where(confidence_condition, consensus_sim, 0.0) # only values above threshold

    # ______Final Score: Match-based F1________
    threshold = min_confidence / num_methods
    f1 = compute_match_f1(consensus_sim, threshold)
    consensus_sim = np.round(consensus_sim.astype(np.float64), 2)
    return consensus_sim, sentences, tasks, f1, method_labels


def visualize_consensus_heatmap(doc_id, method_labels, sentences, tasks, consensus_sim, f1_score):
    # Visualizes the consensus similarity matrix (Jaccard-based) and the Match-based F1 score.
    num_methods = len(method_labels)
    min_confidence = int(num_methods * 2 / 3)
    threshold = min_confidence / num_methods
    fig, ax = plt.subplots(figsize=(14.5, 11))

    #wrap width for sentence and task labels so they fit into window
    wrap_width_s = 50
    wrap_width_t = 30
    labels_s=[]
    for j,s in enumerate(sentences):
        labels_s.append(f"S{j+1}: {textwrap.fill(s,width=wrap_width_s)}")
    labels_t=[]
    for i,t in enumerate(tasks):
        labels_t.append(f"T{i+1}: {textwrap.fill(t,width=wrap_width_t)}")
    df=pd.DataFrame(consensus_sim,index=labels_s,columns=labels_t)

    # colormap and  values below threshold grey
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_under('lightgray')

    # dynamic font size for cells (depends on number of sentences and tasks)
    n_s = len(sentences)
    n_t = len(tasks)
    max_dim = max(n_s, n_t)
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

    # draw heatmap
    sns.heatmap(df,cmap=cmap,annot=True,fmt=".2f",linewidths=.5,ax=ax,vmin=threshold-1e-5,vmax=1.0,annot_kws={"size":font_size})

    # black bounding boxes around the best match for each task column (above threshold)
    best_indices = get_chronological_max_indices(consensus_sim, threshold - 1e-5)
    for col, row in enumerate(best_indices):
        if row != -1: #only for matches above the threshold
            ax.add_patch(plt.Rectangle((col, row), 1, 1,fill=False, edgecolor='black', lw=2))

    title = f"Consensus Matching (Model {doc_id})" if doc_id != "custom" else "Consensus Matching"
    # title and sentence and task label font sizes
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size * 0.7)

    # rotate labels so they are not cutted off
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Display the threshold next to title in the top left corner (like in Threshold_Strategies.py)
    ax.text(0.01, 1.01, f"t={threshold:.2f}",
            transform=ax.transAxes, fontsize=12, color='gray', verticalalignment='bottom', horizontalalignment='left')

    # Display the calculated F1 score in a pink box in the top right corner next to title
    ax.text(0.99, 1.01, f"F1={f1_score:.2f}", transform=ax.transAxes, fontsize=12, color='black',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F2B5DF', edgecolor='gray', alpha=0.9))

    plt.tight_layout()

   #legend box font
    legend_fontsize = max(8, int(font_size * 1.1))
    box_pad = max(0.3, font_size * 0.04)

    #methods and the min confidence
    temp_m=[]
    for m in method_labels:
        temp_m.append(f"  • {m}")
    joined_m="\n".join(temp_m)
    methods_text = ("Used Methods:\n" +joined_m +f"\n\nMin Confidence:\n{min_confidence}/{num_methods} Methods (t={threshold:.2f})")
    fig.text(0.02, 0.02, methods_text, fontsize=legend_fontsize,color='#374151', ha='left',va='bottom',bbox=dict(boxstyle=f'round,pad={box_pad:.2f}', facecolor='#F9FAFB', edgecolor='#E5E7EB', alpha=0.95,lw=1.5))
    plt.show()

#____________________Tupel matching ________________________

def encode_texts(texts, method_config):
    model_name = method_config["embedding"]
    if model_name == "bert":
        return emb.get_bert_embeddings(raw_texts=texts)
    elif model_name == "gemini":
        return emb.get_gemini_embeddings(raw_texts=texts)
    elif model_name == "llm2vec":
        return emb.get_llm2vec_embeddings(raw_texts=texts)


def compute_sim_for_texts(sentence_texts, task_texts, method_config):
    if "embedding" in method_config:
        emb_s = encode_texts(sentence_texts, method_config)
        emb_t = encode_texts(task_texts, method_config)
        return emb.compute_similarity_matrix(method_config["metric"], emb_s, emb_t)
    else:
        return bg.compute_similarity_matrix(
            method_config["traditional"], sentence_texts, task_texts
        )


def build_tuples(items):
    tuples_text = []
    tuples_range = []
    for start in range(len(items) - 1):
        # combines 2 elements of the items list into a string
        combined = ". ".join(items[start: start + 2])
        tuples_text.append(combined)
        # appends the range of indices covered by this tuple to the tuples_range list
        tuples_range.append((start, start + 1))
    return tuples_text, tuples_range


def tuple_matching(data, method_config):
    sentences = data["sentences"]
    tasks = data["tasks"]

    sentence_tuples, sentence_ranges = build_tuples(sentences)
    task_tuples, task_ranges = build_tuples(tasks)
    tuple_sim_matrix = compute_sim_for_texts(sentence_tuples, task_tuples, method_config)

    return tuple_sim_matrix, sentence_tuples, task_tuples, sentence_ranges, task_ranges


def visualize_tuple_heatmap(doc_id, sim_matrix, sentence_ranges, task_ranges, sentences, tasks, method_config, STRATEGY,LEMMATIZE, REMOVE_COND, title_suffix=""):
    # labels with custom task labels: e.g. "T1: ... . T2: ..."
    col_labels=[]
    for t_range in task_ranges:
        temp_t=[]
        for g in range(t_range[0],t_range[1]+1):
            temp_t.append(f"T{g+1}: {tasks[g]}")
        col_labels.append(". ".join(temp_t))

    row_labels = []
    for s_range in sentence_ranges:
        first_s = sentences[s_range[0]]
        rest_s = []
        for g in range(s_range[0] + 1, s_range[1] + 1):
            rest_s.append(f"S{g + 1}: {sentences[g]}")
        row_labels.append(". ".join([first_s] + rest_s))

    fig, ax = plt.subplots(figsize=(14.5, 11))
    title = f"Tuple Matching (Model {doc_id}) {title_suffix}" if doc_id != "custom" else f"Tuple Matching {title_suffix}"
    threshold = get_thresholds(method_config, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    ts.draw_single_basic_heatmap(ax, sim_matrix, row_labels, col_labels, threshold, title, custom=True)
    plt.tight_layout()
    plt.show()

# ________________________Best-Of-Tuple Matching________________________

def compute_merged_similarity(sentence_index, indices, sentences, tasks, method_config, emb_sentences=None):
    # similarity between a sentence and a merged task group
    temp_m=[]  #temporary list of tasks to be merged
    for g in indices:
        temp_m.append(tasks[g])
    merged_text=". ".join(temp_m)
    if "embedding" in method_config:
        merged_emb = np.array(encode_texts([merged_text], method_config))
        return emb.compute_similarity_matrix(
            method_config["metric"], emb_sentences[sentence_index:sentence_index + 1], merged_emb)[0, 0] #[0,0] to extract the similarity value from the 1x1 matrix
    else:
        return compute_sim_for_texts([sentences[sentence_index]], [merged_text], method_config)[0, 0]


def best_of_tuple_matching(data, method_config):
    sentences = data["sentences"]
    tasks = data["tasks"]
    n_tasks = len(tasks)

    # base/normal similarity matrix
    emb_sentences = None
    if "embedding" in method_config:
        emb_sentences = np.array(encode_texts(sentences, method_config))
        base_sim = emb.compute_similarity_matrix(method_config["metric"],emb_sentences, np.array(encode_texts(tasks, method_config)))
    else:
        base_sim = compute_sim_for_texts(sentences, tasks, method_config)

    # for each sentence, merge consecutive tasks
    # if sim(merged, s) > max(sim(current_group, s), sim(next_task, s)) 
    all_groupings = []
    all_group_sims = []
    for sj in range(len(sentences)):
        groups = []
        group_sims = []
        i = 0
        while i < n_tasks:
            current = [i]
            cur_sim = base_sim[sj, i]

            #  extending the group with the next task
            while i + len(current) < n_tasks:
                next_index = i + len(current)
                merged_sim = compute_merged_similarity(sj, current + [next_index], sentences,tasks, method_config,emb_sentences)
                next_task_sim = base_sim[sj, next_index]

                # merge only if merged similarity increases and exceeds next individuel similiarity
                if merged_sim > max(cur_sim, next_task_sim):
                    current.append(next_index)
                    cur_sim =merged_sim
                else:
                    break

            groups.append(current)
            group_sims.append(cur_sim)
            # move to the next task after the current group
            i += len(current)

        all_groupings.append(groups)
        all_group_sims.append(group_sims)

    # selection of best non overlapping groups
    # collect all candidate groups with their similarity values
    candidates = []

    # merged groups from per-sentence merging
    for sj, (grps, sims) in enumerate(zip(all_groupings, all_group_sims)):
        for grp, sim in zip(grps, sims):
            candidates.append((grp, sim, sj))

    
    # subgroups, so other sentences still have a chance to match with them
    sub_groups = {}
    for grps in all_groupings:
        for grp in grps:
            for length in range(1, len(grp)):
                for i in range(len(grp) - length + 1):
                    sub = tuple(grp[i:i + length])
                    # Join the text of the tasks in this sub-group
                    temp_s=[]
                    for g in sub:
                        temp_s.append(tasks[g])
                    sub_groups[sub]=". ".join(temp_s)

    if sub_groups:
        sub_grps=[]
        for k in sub_groups.keys(): sub_grps.append(list(k))
        sub_texts=list(sub_groups.values())

        # compute similarity for all sub-groups against all sentences
        if "embedding" in method_config and emb_sentences is not None:
            sub_embs = np.array(encode_texts(sub_texts, method_config))
            sub_sims = emb.compute_similarity_matrix(method_config["metric"], emb_sentences, sub_embs)
            
        else:
            sub_sims = compute_sim_for_texts(sentences, sub_texts, method_config)

        # add sub-group similarities to candidates list
        for index, sub_grp in enumerate(sub_grps):
            for si in range(len(sentences)):
                candidates.append((sub_grp, sub_sims[si, index], si))

    # sort candidates by similarity, pick non overlapping groups
    candidates.sort(key=lambda x: x[1], reverse=True)
    used_tasks = set()
    groups = []
    for grp, sim, si in candidates:
        if set(grp).isdisjoint(used_tasks):
            groups.append(grp) 
            used_tasks.update(grp) # add the used tasks
        if len(used_tasks) == n_tasks:  # stop once all tasks are covered
            break

    # add any remaining uncovered tasks as individual groups
    for t in range(n_tasks):
        if t not in used_tasks:
            groups.append([t])

    # sort groups by the first task index for correctt column ordering
    groups.sort(key=lambda g: g[0])

    # build final similarity matrix (all sentences vs selected task groups)
    group_texts=[]
    for grp in groups:
        temp_g=[]
        for g in grp:
            temp_g.append(tasks[g])
        g_text=". ".join(temp_g)
        group_texts.append(g_text)
    sim_matrix=compute_sim_for_texts(sentences,group_texts,method_config)

    return sim_matrix, groups


def visualize_best_of_tuple_heatmap(doc_id, sim_matrix, combo_source, sentences, tasks, method_config):
    # build labels with cutom task labels: e.g. "T1: ... . T2: ..."
    col_labels=[]
    for grp in combo_source:
        temp_c=[]
        for g in grp:
            temp_c.append(f"T{g+1}: {tasks[g]}")
        col_labels.append(". ".join(temp_c))

    fig, ax = plt.subplots(figsize=(14.5, 11))
    threshold = get_thresholds(method_config, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    title = f"Best-Of-Tuple Matching (Model {doc_id})" if doc_id != "custom" else "Best-Of-Tuple Matching"
    ts.draw_single_basic_heatmap(ax, sim_matrix, sentences,col_labels, threshold, title, custom=True)
    plt.tight_layout()
    plt.show()


#get Threshold for all methods
def get_thresholds(method_config, strategy=1, lemmatize=False, remove_cond=False):
    return ts.get_precomputed_threshold(method_config, strategy, lemmatize, remove_cond)


if __name__ == "__main__":

    method_cfg = METHOD_CONFIG

    # 1. TEST CONSENSUS MATCHING
    if RUN_CONSENSUS:
        print(f"Consensus Matching for Model {DOC_ID}___")
        consensus_sim, sentences, tasks, f1, method_labels = consensus_matching(
            DOC_ID, CONSENSUS_METHODS, strategy=STRATEGY, lemmatize=LEMMATIZE,
            remove_cond=REMOVE_COND, text=TEXT, bpmn_xml=BPMN_XML
        )
        visualize_consensus_heatmap(
            DOC_ID, method_labels, sentences, tasks, consensus_sim, f1
        )
      
    # load data for the following methods
    if RUN_TUPLE or RUN_BEST_OF_TUPLE:
        data_dict = ts.load_data(method_cfg, [DOC_ID], LEMMATIZE, REMOVE_COND, text=TEXT, bpmn_xml=BPMN_XML)
        data = data_dict[DOC_ID]

    # 2. TEST TUPLE MATCHING
    if RUN_TUPLE:
        print(f"Tuple Matching (size=2) for Model {DOC_ID}____")
        sim_tuple, s_tuples, t_tuples, s_ranges, t_ranges = tuple_matching(
            data, method_cfg
        )
        visualize_tuple_heatmap(
            DOC_ID, sim_tuple, s_ranges, t_ranges, data["sentences"], data["tasks"], method_config=method_cfg,
            STRATEGY=STRATEGY, LEMMATIZE=LEMMATIZE, REMOVE_COND=REMOVE_COND
        )

    # 3. TEST BEST-OF-TUPLE MATCHING
    if RUN_BEST_OF_TUPLE:
        print(f"Best-Of-Tuple Matching for Model {DOC_ID}____")
        sim_best, combo_src = best_of_tuple_matching(data, method_cfg)
        visualize_best_of_tuple_heatmap(
            DOC_ID, sim_best, combo_src, data["sentences"], data["tasks"],
            method_config=method_cfg
        )
