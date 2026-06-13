import os
import time
import pickle
import logging

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity,laplacian_kernel,rbf_kernel
import Datasets
from sentence_transformers import SentenceTransformer
from google import genai
import numpy as np
from dotenv import load_dotenv
import warnings
import textwrap

#Suppress transformers Warning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
#suppress Already found a peft_config warning
warnings.filterwarnings("ignore", category=UserWarning, module="peft")

#suppress tokenizer message from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

cached_models = {}

def preload_models():
    if "bert" not in cached_models:
        cached_models["bert"] = SentenceTransformer('stsb-roberta-large')
    if "llm2vec" not in cached_models:
        from llm2vec import LLM2Vec
        cached_models["llm2vec"] = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
    if "gemini" not in cached_models:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            cached_models["gemini"] = genai.Client(api_key=api_key)

def get_bert_embeddings(doc_ids=None, lemmatize_data=False, remove_conditions=False, load_stored=True, text=None, bpmn_xml=None, raw_texts=None): # https://huggingface.co/sentence-transformers/stsb-roberta-large
    #if raw_texts is provided, encode them directly and return embeddings (used by Tuple/Best-Of-Tuple Matching)
    if raw_texts is not None:
        if "bert" not in cached_models:
            cached_models["bert"] = SentenceTransformer('stsb-roberta-large')
        return cached_models["bert"].encode(raw_texts)
    # load already computed embeddings
    if load_stored and text is None:
        #print(f"Loading stored data for BERT embeddings for models: {doc_ids}")
        data = {}
        for d in doc_ids:
            p = f"embeddings/bert_{d}_L{lemmatize_data}_R{remove_conditions}.pkl"
            if os.path.exists(p):
                with open(p, "rb") as f: 
                    data[d]=pickle.load(f)
        if len(data) ==len(doc_ids):
             return data
    #print("Computing BERT embeddings")
    if "bert" not in cached_models:
        cached_models["bert"] = SentenceTransformer('stsb-roberta-large')
    st_model = cached_models["bert"]
    data = {}
    for doc_id in doc_ids:
        tasks, sentences = Datasets.get_data(doc_id=doc_id, text=text, bpmn_xml=bpmn_xml, lemmatized=lemmatize_data, remove_conditions=remove_conditions)

        emb_s =st_model.encode(sentences)
        emb_t =st_model.encode(tasks)

        data[doc_id] = {"sentences": sentences, "tasks": tasks,"emb_sentences": emb_s, "emb_tasks":emb_t}
    return data

def get_llm2vec_embeddings(doc_ids=None, lemmatize_data=False, remove_conditions=False, load_stored=True, text=None, bpmn_xml=None, raw_texts=None):
    #if raw_texts is provided, encode them directly and return embeddings (used by Tuple/Best-Of-Tuple Matching)
    if raw_texts is not None:
        if "llm2vec" not in cached_models:
            from llm2vec import LLM2Vec #https://pypi.org/project/llm2vec/0.1.4/ 
            cached_models["llm2vec"] = LLM2Vec.from_pretrained(
                "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", # https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised
                peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
        return cached_models["llm2vec"].encode(raw_texts, show_progress_bar=False)
    if load_stored and text is None:
        #print(f"Loading stored data for LLM2Vec embeddings for models: {doc_ids}")
        data = {}
        for d in doc_ids:
            p = f"embeddings/llm2vec_{d}_L{lemmatize_data}_R{remove_conditions}.pkl"
            if os.path.exists(p):
                with open(p, "rb") as f: 
                    data[d] = pickle.load(f)
        if len(data) == len(doc_ids): 
            return data
    #print(f"Computing LLM2Vec embeddings for models: {doc_ids}")
    if "llm2vec" not in cached_models:
        from llm2vec import LLM2Vec    
        cached_models["llm2vec"] = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
    l2v = cached_models["llm2vec"]

    data = {}
    for doc_id in doc_ids:
        tasks, sentences = Datasets.get_data(doc_id=doc_id, text=text, bpmn_xml=bpmn_xml, lemmatized=lemmatize_data, remove_conditions=remove_conditions)

        emb_s = l2v.encode(sentences, show_progress_bar=False)
        emb_t = l2v.encode(tasks, show_progress_bar=False)

        data[doc_id] = {
            "sentences": sentences, "tasks": tasks,
            "emb_sentences": emb_s, "emb_tasks": emb_t
        }
    return data

def get_gemini_embeddings(doc_ids=None, lemmatize_data=False,remove_conditions=False, model_name="gemini-embedding-2", load_stored=True, text=None, bpmn_xml=None, raw_texts=None):
    #if raw_texts is provided, encode them directly and return embeddings (used by Tuple/Best-Of-Tuple Matching)
    if raw_texts is not None:
        if "gemini" not in cached_models:
            load_dotenv()
            cached_models["gemini"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        client = cached_models["gemini"]
        embs = []
        rate_limit_count = 0
        for t in raw_texts:
            while True:
                try:
                    res = client.models.embed_content(model="gemini-embedding-2", contents=t)
                    embs.append(res.embeddings[0].values)
                    rate_limit_count = 0
                    break
                except Exception as e:
                    if "429" in str(e):
                        rate_limit_count += 1
                        if rate_limit_count >= 5:
                            raise Exception("Daily quota reached for Gemini Embeddings")
                        time.sleep(20)
        return np.array(embs)
    if load_stored and text is None:
        #print(f"Loading stored data for Gemini embeddings for models: {doc_ids}")
        data = {}
        for d in doc_ids:
            p = f"embeddings/gemini_{d}_L{lemmatize_data}_R{remove_conditions}.pkl"
            if os.path.exists(p):
                with open(p, "rb") as f:
                     data[d] = pickle.load(f)
        if len(data) == len(doc_ids): 
            return data
    #print(f"Computing Gemini embeddings for models: {doc_ids}")
    load_dotenv() # read api key from .env
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    data = {}
    rate_limit_count = 0
    for doc_id in doc_ids:
        tasks, sentences = Datasets.get_data(doc_id=doc_id, text=text, bpmn_xml=bpmn_xml, lemmatized=lemmatize_data, remove_conditions=remove_conditions)

        # Embedding for sentences
        emb_s = []
        for sentence in sentences:
            while True:
                try:
                    res = client.models.embed_content(model=model_name, contents=sentence)
                    emb_s.append(res.embeddings[0].values)
                    rate_limit_count = 0
                    break
                except Exception as e: #handle rate limits
                    if "429" in str(e) :
                        rate_limit_count += 1
                        if rate_limit_count>= 5: 
                            raise Exception("Daily quota reached for Gemini Embeddings") 
                        print(f"  Rate limit hit for Model {doc_id} (sentence). Waiting 20s")
                        time.sleep(20)
        # Embedding for tasks
        emb_t = []
        for task in tasks:
            while True:
                try:
                    res = client.models.embed_content(model=model_name, contents=task)
                    emb_t.append(res.embeddings[0].values)
                    rate_limit_count = 0
                    break
                except Exception as e:
                    if "429" in str(e):
                        rate_limit_count += 1
                        if rate_limit_count>= 5:
                            raise Exception("Daily quota reached for Gemini Embeddings")
                        print(f"  Rate limit hit for Model {doc_id} (task). Waiting 20s")
                        time.sleep(20)
                    
        data[doc_id] = {
            "sentences": sentences, "tasks": tasks,
            "emb_sentences": emb_s, "emb_tasks": emb_t
        }
    return data

def save_embeddings_to_file(data, method, lem, rem):
    os.makedirs("../Bachelor_projekt/embeddings", exist_ok=True)
    for d, v in data.items():
        with open(f"embeddings/{method}_{d}_L{lem}_R{rem}.pkl", "wb") as f:
            pickle.dump(v, f)

def generate_and_save_all_embeddings(doc_ids):
    for l in [True, False]:
        for r in [True, False]:
            save_embeddings_to_file(get_bert_embeddings(doc_ids, l, r, False), "bert", l, r)
            save_embeddings_to_file(get_llm2vec_embeddings(doc_ids, l, r, False), "llm2vec", l, r)
            save_embeddings_to_file(get_gemini_embeddings(doc_ids, l, r, load_stored=False), "gemini", l, r)


#_________________________compute similarity matrix________________________________________________________
def compute_similarity_matrix(metric, emb_s, emb_t):
    if metric == "cos":
        return np.round(cosine_similarity(emb_s, emb_t).astype(np.float64), 2)#accurate roudning
    elif metric == "man":
        return np.round(laplacian_kernel(emb_s, emb_t).astype(np.float64), 2)
    elif metric == "eu":
        return np.round(rbf_kernel(emb_s, emb_t).astype(np.float64), 2)


#________________________plot________________________________________________

def plot_basic_heatmap(similarity, sentences, tasks, threshold, title="Similarity Heatmap"):
    import Threshold_Strategies as ts
    fig, ax = plt.subplots(figsize=(14.5, 11))
    ts.draw_single_basic_heatmap(ax, similarity, sentences, tasks, threshold, title)
    plt.tight_layout()
    plt.show()

#without threshold
def plot_similarity_heatmap(data, embedding_method, metric, title_prefix="Similarity Heatmap"):
    for doc_id, data_entry in data.items():
        emb_s = data_entry["emb_sentences"]
        emb_t = data_entry["emb_tasks"]
        sentences = data_entry["sentences"]
        tasks = data_entry["tasks"]

        sim_matrix = compute_similarity_matrix(metric, emb_s, emb_t)

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


def get_threshold(embedding_method, metric, strategy=1, lemmatize=False, remove_cond=False):
    import Threshold_Strategies as ts
    method_config = {"embedding": embedding_method, "metric": metric}
    return ts.get_precomputed_threshold(method_config, strategy, lemmatize, remove_cond)

if __name__ == "__main__":
    DOCS = ["01"]
    #generate_and_save_all_embeddings(DOCS)


    LEMMATIZE_DATA = False
    REMOVE_CONDITIONS = False
    
    EMBEDDING_METHOD = "llm2vec"
    METRIC = "man"

    TEXT = "The customer places an order. We receive the order and process the payment. Finally, the goods are shipped to the customer."
    
    BPMN_XML =  """<testset xmlns="http://cpee.org/ns/properties/2.0">
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

    if EMBEDDING_METHOD == "bert":
        data_bert = get_bert_embeddings(DOCS, lemmatize_data=LEMMATIZE_DATA, remove_conditions=REMOVE_CONDITIONS, text=TEXT, bpmn_xml=BPMN_XML)
        
        first_doc_id = list(data_bert.keys())[0]
        sim_matrix = compute_similarity_matrix(METRIC, data_bert[first_doc_id]["emb_sentences"], data_bert[first_doc_id]["emb_tasks"])
        threshold = get_threshold("bert", METRIC, strategy=1,lemmatize=LEMMATIZE_DATA, remove_cond=REMOVE_CONDITIONS)
        title = f"BERT Embeddings + {METRIC.upper()}" + (f" - Doc: {first_doc_id}" if first_doc_id != "custom" else "")
        plot_basic_heatmap(sim_matrix, data_bert[first_doc_id]["sentences"], data_bert[first_doc_id]["tasks"],threshold, title)

        #plot_similarity_heatmap(data_bert, EMBEDDING_METHOD, METRIC, title_prefix=f"BERT Embeddings + {METRIC}")

    elif EMBEDDING_METHOD == "gemini":
        data_gemini = get_gemini_embeddings(DOCS, lemmatize_data=LEMMATIZE_DATA, remove_conditions=REMOVE_CONDITIONS, text=TEXT, bpmn_xml=BPMN_XML)
        #plot_similarity_heatmap(data_gemini, "gemini", METRIC, title_prefix="Gemini embeddings")
        if data_gemini:
            first_doc_id = list(data_gemini.keys())[0]
            sim_matrix = compute_similarity_matrix(METRIC, data_gemini[first_doc_id]["emb_sentences"], data_gemini[first_doc_id]["emb_tasks"])
            threshold = get_threshold("gemini", METRIC, strategy=1,lemmatize=LEMMATIZE_DATA, remove_cond=REMOVE_CONDITIONS)
            plot_basic_heatmap(sim_matrix, data_gemini[first_doc_id]["sentences"], data_gemini[first_doc_id]["tasks"], threshold, f"Gemini Embeddings - Doc: {first_doc_id}")
            #plot_similarity_heatmap(data_gemini, EMBEDDING_METHOD, METRIC, title_prefix="Gemini Embeddings")

    elif EMBEDDING_METHOD == "llm2vec":
        data_l2v = get_llm2vec_embeddings(DOCS, lemmatize_data=LEMMATIZE_DATA, remove_conditions=REMOVE_CONDITIONS, text=TEXT, bpmn_xml=BPMN_XML)
        #plot_similarity_heatmap(data_l2v, "llm2vec", METRIC, title_prefix=f"LLM2Vec embeddings + {METRIC}")
        if data_l2v:
            first_doc_id = list(data_l2v.keys())[0]
            sim_matrix = compute_similarity_matrix(METRIC, data_l2v[first_doc_id]["emb_sentences"], data_l2v[first_doc_id]["emb_tasks"])
            threshold = get_threshold("llm2vec", METRIC, strategy=1,lemmatize=LEMMATIZE_DATA, remove_cond=REMOVE_CONDITIONS)
            plot_basic_heatmap(sim_matrix, data_l2v[first_doc_id]["sentences"], data_l2v[first_doc_id]["tasks"], threshold, f"LLM2Vec Embeddings + {METRIC} - Doc: {first_doc_id}")
            #plot_similarity_heatmap(data_l2v, "llm2vec", METRIC, title_prefix=f"LLM2Vec Embeddings + {METRIC.upper()}")



    