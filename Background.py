import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


#___________________________syntactic similarity methods_________________________

# Levenshtein distance
from rapidfuzz.distance import Levenshtein

def Levenshtein_similarity(x, y):
    words_x = x.lower().split()
    words_y = y.lower().split()
        
    # how well words in x are covered by words in y
    scores_x = []
    for wx in words_x:
        best_score = 0.0
        for wy in words_y:
            score = Levenshtein.normalized_similarity(wx, wy)
            if score > best_score:
                best_score = score
        scores_x.append(best_score)
        
    avg_x = sum(scores_x) / len(scores_x)
    
    # how well words in y are covered by words in x
    scores_y = []
    for wy in words_y:
        best_score = 0.0
        for wx in words_x:
            score = Levenshtein.normalized_similarity(wx, wy)
            if score > best_score:
                best_score = score
        scores_y.append(best_score)
        
    avg_y = sum(scores_y) / len(scores_y)
        
    return round((avg_x + avg_y) / 2, 2)


# Jaccard, Source: https://www.newscatcherapi.com/blog-posts/ultimate-guide-to-text-similarity-with-python
def Jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return round(intersection_cardinality / float(union_cardinality), 2)


#___________________________knowledge based similarity methods______________________________________

import nltk  # https://www.nltk.org
from nltk.corpus import wordnet as wn  # https://www.nltk.org/howto/wordnet.html

# nltk.download('punkt', quiet=True) #Tokenization rules for word_tokenize
# nltk.download('wordnet', quiet=True) #WordNet Database
# nltk.download('punkt_tab', quiet=True) # new format for punkt

# Converts a sentence into a list of WordNet synsets
def to_synsets(sentence):
    words = sentence.lower().split()

    synsets = []
    for word in words:
        # look up the word in WordNet, returns a list of possible meanings
        word_synsets = wn.synsets(word)

        # if the word exists in WordNet, take the first(= most common) meaning
        if word_synsets:
            synsets.append(word_synsets[0])

    return synsets


# For each word in src(source), finds the most similar word in tgt(target)
# then returns the average of all those best-match scores
# 1. sent1 -> sent2     2. (sent2 -> sent1)
def best_match_avg(src, tgt):
    scores = []
    for syn_a in src:
        sims = []
        for syn_b in tgt:
            # path_similarity compares two synsets based on their distance in the WordNet hierarchy
            # --> see formula in Thesis
            sim = syn_a.path_similarity(syn_b)
            if sim is not None:
                sims.append(sim)

        # keep only the highest score (=best matching word in tgt)
        if sims:
            scores.append(max(sims))

    # average the best-match scores across all words in src
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0


def WordNet_similarity(s1, s2):
    syns1 = to_synsets(s1)
    syns2 = to_synsets(s2)
    # compare in both directions because best-match is not symmetric
    s1_to_s2 = best_match_avg(syns1, syns2)  # how well s1 is covered by s2
    s2_to_s1 = best_match_avg(syns2, syns1)  # how well s2 is covered by s1

    return round((s1_to_s2 + s2_to_s1) / 2, 2)


#___________________________corpus based similarity methods______________________

# non-machine-learning (TF-IDF)
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

def TF_IDF_similarity(s1, s2):
    vectorizer = TfidfVectorizer()

    # convert both sentences into TF-IDF vectors at once
    tfidf_matrix = vectorizer.fit_transform([s1, s2])

    # tfidf_matrix[0] = vector of s1, tfidf_matrix[1] = vector of s2
    # result is a 2D matrix e.g. [[0.57]] -> need [0][0] to extract the single value
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return round(float(score[0][0]), 2)


# machine-learning (Word2Vec)
import gensim.downloader as api
# https://radimrehurek.com/gensim/models/keyedvectors.html

# Cache the model to avoid re-loading it every time
word2vec_model = None

def getword2vec_model():
    global word2vec_model
    if word2vec_model is None:
        #print("Loading Word2Vec model")
        word2vec_model =api.load("word2vec-google-news-300")
    return word2vec_model

#Converts a sentence into a single vector by averaging all word vectors
def sentence_vector(sentence, model):
    words = sentence.lower().split()
    known_vectors = []
    for w in words:
        if w in model:
            known_vectors.append(model[w])
    if not known_vectors:
        return np.zeros(model.vector_size)
    return np.mean(known_vectors, axis=0)


def Word2Vec_similarity(s1, s2):
    model = getword2vec_model()
    vec1 = sentence_vector(s1, model).reshape(1, -1)
    vec2 = sentence_vector(s2, model).reshape(1, -1)
    score = cosine_similarity(vec1, vec2)
    return round(float(score[0][0]), 2)


#_____________compute similarity matrix ______________________________________

def compute_similarity_matrix(traditional_method, sentences, tasks):
    n_s, n_t = len(sentences), len(tasks)
    sim_matrix = np.zeros((n_s, n_t))

    if traditional_method == "tfidf":
        for j, s in enumerate(sentences):
            for i, t in enumerate(tasks):
                sim_matrix[j, i] = TF_IDF_similarity(s, t)
    elif traditional_method == "wordnet":
        for j, s in enumerate(sentences):
            for i, t in enumerate(tasks):
                sim_matrix[j, i] = WordNet_similarity(s, t)
    elif traditional_method == "levenshtein":
        for j, s in enumerate(sentences):
            for i, t in enumerate(tasks):
                sim_matrix[j, i] = Levenshtein_similarity(s, t)
    elif traditional_method == "jaccard":
        for j, s in enumerate(sentences):
            for i, t in enumerate(tasks):
                sim_matrix[j, i] = Jaccard_similarity(s.split(), t.split())
    elif traditional_method == "word2vec":
        for j, s in enumerate(sentences):
            for i, t in enumerate(tasks):
                sim_matrix[j, i] = Word2Vec_similarity(s, t)

    return np.round(sim_matrix.astype(np.float64), 2)#accurate roudning



#______________________- Precision / Recall / F1  (Match-based)


def get_asymmetric_diagonal_row(col_index, num_rows, num_cols):
  
   # Maps a column index to the diagonal GT row index
    if num_cols <= 1:
        return 0
    if num_rows <= 1:
        return 0
    return int(round(col_index * (num_rows - 1) / (num_cols - 1)))


def get_chronological_max_indices(matrix, threshold):
    rows, cols = matrix.shape
    best_indices = []

    for i in range(cols):
        #maximum similarity in this column(task)
        max_val = np.max(matrix[:, i])
        if max_val < threshold:
            best_indices.append(-1)
            continue

        #find all row indices (sentences) that share this same maximum similarity
        max_indices = np.where(matrix[:, i] == max_val)[0]

        expected_j = get_asymmetric_diagonal_row(i, rows, cols)

        distances = np.abs(max_indices - expected_j)

        # np.argmin(distances) returns the position (index) of the smallest value in the distances array
        best_j = max_indices[np.argmin(distances)]
        best_indices.append(best_j)

    return np.array(best_indices)


def match_precision(matrix, threshold=0.5):
    #Match-based Precision
    
    num_tasks = matrix.shape[1]
    
   
    #np.any(.., axis=0) checks if at least one True exists in each column
    #.sum counts how many tasks have a valid match
    match_count = np.any(matrix >= threshold, axis=0).sum()
    return match_count / num_tasks if num_tasks > 0 else 0.0


def match_recall(matrix, threshold):
    #Match-based Recall
    num_rows = matrix.shape[0]
    # best row index for each task
    best_indices = get_chronological_max_indices(matrix, threshold)
    
    # ignore columns that returned -1 (no match)
    valid_indices = best_indices[best_indices != -1]
    
    #count how many sentences are linked to at least one task
    # set() ensures unique rows are counted (e.g., if two tasks match the same sentence)
    match_count = len(set(valid_indices))
    
    # Return ratio of unique matched sentences to total number of sentences
    return match_count / num_rows if num_rows > 0 else 0.0


def f1_score(p, r):
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def compute_match_f1(matrix, threshold):
    p = match_precision(matrix, threshold)
    r = match_recall(matrix, threshold)
    return f1_score(p, r)

#___________________

#return the best threshold for a given method 
def get_threshold(method_info, strategy=1, lemmatize=False, remove_cond=False):
    import Threshold_Strategies as ts
    method_config = {"traditional": method_info}
    return ts.get_precomputed_threshold(method_config, strategy, lemmatize, remove_cond)

#_________________________plot__________________________________--

def plot_basic_heatmap(similarity, sentences, tasks, threshold, title="Similarity Heatmap"):
    import Threshold_Strategies as ts
    fig, ax = plt.subplots(figsize=(14.5, 11))
    ts.draw_single_basic_heatmap(ax, similarity, sentences, tasks, threshold, title)
    plt.tight_layout()
    plt.show()


#without threshold
def plot_similarity_heatmap(data, traditional_method, title_prefix="Similarity Heatmap"):
    for doc_id, data_entry in data.items():
        sentences =data_entry["sentences"]
        tasks=data_entry["tasks"]

        sim_matrix = compute_similarity_matrix(traditional_method, sentences, tasks)

        #ensure negative values are set to 0
        sim_matrix[sim_matrix < 0] = 0

        plt.figure(figsize=(14.5, 11))
        #wrap width for sentence and task labels so they fit into window
        wrap_width_s = 50
        wrap_width_t = 30
        labels_s=[]
        for j,s in enumerate(sentences):
            labels_s.append(f"S{j+1}: {textwrap.fill(s,width=wrap_width_s)}")
        labels_t=[]
        for i,t in enumerate(tasks):
            labels_t.append(f"T{i+1}: {textwrap.fill(t,width=wrap_width_t)}")

        #dynamic font size for cells (depends on number of sentences and tasks)
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

        #draw heatmap
        sns.heatmap(sim_matrix,annot=True,cmap="YlGnBu",xticklabels=labels_t,yticklabels=labels_s,annot_kws={"size":font_size})
        
        title = f"{title_prefix} - Doc: {doc_id}" if doc_id != "custom" else title_prefix
        plt.title(title, fontsize=18, fontweight='bold')
        
        #rotate labels so they are not cutted off
        plt.xticks(rotation=45, ha='right', fontsize=font_size*0.7)
        plt.yticks(rotation=0, fontsize=font_size*0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import Datasets
    
    DOC_ID = "21"
    METHOD = "levenshtein"
    STRATEGY = 1
    LEMMATIZE = True
    REMOVE_COND = False

   
    TEXT = "The customer places an order. We receive the order and process the payment. Finally, the goods are shipped to the customer."
    
    BPMN_XML ="""<testset xmlns="http://cpee.org/ns/properties/2.0">
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

    
    tasks, sentences = Datasets.get_data(doc_id=DOC_ID, text=TEXT, bpmn_xml=BPMN_XML,
                                          lemmatized=LEMMATIZE, remove_conditions=REMOVE_COND)
    sim_matrix = compute_similarity_matrix(METHOD, sentences, tasks)
    threshold = get_threshold(METHOD, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    print(threshold)
    print("threshold = " + str(threshold))
    plot_basic_heatmap(sim_matrix, sentences, tasks, threshold, title=f"Heatmap ({METHOD.upper()})")
    
    """
    DOC_ID = [DOC_ID]
    data = {}
    for d in DOC_ID:
        tasks, sentences = Datasets.get_data(doc_id=d, lemmatized=LEMMATIZE, remove_conditions=REMOVE_COND)
        data[d] = {"sentences": sentences, "tasks": tasks}
        
    #plot_similarity_heatmap(data, METHOD, title_prefix=f"{METHOD.upper()}")
    sim_matrix = compute_similarity_matrix(METHOD, data[DOC_ID[0]]["sentences"], data[DOC_ID[0]]["tasks"])
    threshold = get_threshold(METHOD, strategy=STRATEGY, lemmatize=LEMMATIZE, remove_cond=REMOVE_COND)
    plot_basic_heatmap(sim_matrix, data[DOC_ID[0]]["sentences"], data[DOC_ID[0]]["tasks"], threshold, title=f"Heatmap ({METHOD.upper()})")
"""
    
