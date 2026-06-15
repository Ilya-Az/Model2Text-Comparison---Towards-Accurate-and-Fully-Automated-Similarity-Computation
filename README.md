Model2Text Comparison – Towards Accurate and Fully Automated Similarity Computation
A fully automated pipeline for computing the similarity between BPMN process models and natural language process descriptions. Developed as a Bachelor's Thesis at the Chair of Information Systems and Business Process Management (i17), TU Munich.
Overview
The system decomposes a BPMN model and a textual description into atomic units (tasks and sentences), computes pairwise similarity scores using a range of text similarity methods, applies an optimized threshold to classify matches, and aggregates results into a single F1-based Model2Text similarity score.
Features
Multiple text similarity methods: Levenshtein, Jaccard, WordNet, TF-IDF, Word2Vec, BERT, LLM2Vec, Gemini Embeddings
Three threshold optimization strategies (F1-Gap, Diagonal GT, True GT)
Further dimension approaches: Tuple Matching, Best-Of-Tuple Matching, Consensus Matching
Optional preprocessing: Lemmatization, Relevant Clause Extraction (RCE)
Heatmap visualization of sentence-task similarity matrices
REST API via Flask as an Interface for the Interagrtion in external applications


Installation
1. Clone the repository
git clone https://github.com/Ilya-Az/Model2Text-Comparison---Towards-Accurate-and-Fully-Automated-Similarity-Computation.git

2. Install dependencies
pip install numpy scikit-learn matplotlib seaborn pandas
pip install rapidfuzz nltk gensim spacy flask
pip install sentence-transformers
pip install torch transformers peft==0.11.1 llm2vec
pip install google-genai python-dotenv

⚠️ llm2vec requires exactly peft==0.11.1. Other versions may cause compatibility issues.
3. Download the spaCy language model
python3 -m spacy download en_core_web_sm

4. Download required NLTK packages
Run the following once in a Python shell or script before first use:
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

5. Set up your Gemini API key


Create a .env file in the root directory:
GEMINI_API_KEY=your_api_key_here

The .env file is listed in .gitignore and will not be pushed to GitHub.

Usage
Run the API server
python3 api.py
The Flask server starts locally. Send a POST request with the following JSON body:
{
  "similarity_panel": True,
   "text": "Your process description here...",
   "bpmn_xml": <your BPMN/CPEE XML string>,
    "approach": "Insert Further Dimension Approach",
    "methods": [
           method1, method2, …
       ],
     "lemmatize":Input is Boolean,
     "remove_cond": Input is Boolean,
}

List of Text Similarity Methods for Input in JSON body
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


Text Similarity Methods by Type
Method
Type
Levenshtein
Syntactic
Jaccard
Syntactic
WordNet
Knowledge-based
TF-IDF
Corpus-based
Word2Vec
Corpus-based
BERT (stsb-roberta-large)
Contextual Embedding
LLM2Vec (Sheared-LLaMA)
Contextual Embedding
Gemini Embeddings (gemini-embedding-2)
Contextual Embedding


List of Further Dimension Approaches for Input in JSON body
   APPROACHES = ["tuple", "best_of_tuple", "consensus"]

Project Structure
├── Datasets Folder    # stores the Datasets of the PMo Dataset which is used in this work
├── embedding Folder   #stores the precomputed embeddings for the Datasets
├── datasets.py             # Task & sentence extraction, lemmatization, RCE, GT data
├── background.py           # Traditional similarity methods & Match-based F1
├── new_and_state_of_the_art.py  # BERT, LLM2Vec, Gemini Embeddings
├── threshold_strategies.py      # Three threshold optimization strategies
├── further_dimension.py         # Tuple Matching, Best-Of-Tuple, Consensus Matching
├── autobpmn_ai_service.py       # Facade layer for AutoBPMN.AI
├── api.py                 # Flask REST API
├── evaluation.py             # Spearman correlation, Jaccard Index, GT-F1 evaluation
├── .env                     # Gemini API key (not tracked by git)
└── README.md



