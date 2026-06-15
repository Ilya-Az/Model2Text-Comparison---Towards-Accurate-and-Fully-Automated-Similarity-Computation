# Model2Text Comparison - Towards Accurate and Fully Automated Similarity Computation

A fully automated pipeline for computing the similarity between BPMN process models and natural language process descriptions. Developed as a Bachelor's Thesis at the Chair of Information Systems and Business Process Management (i17), TU Munich.



## Overview

The system decomposes a BPMN model and a textual description into atomic units (tasks and sentences), computes pairwise similarity scores using a range of text similarity methods, applies an optimized threshold to classify matches, and aggregates the results into a single F1-based Model2Text similarity score.



## Features

- Multiple text similarity methods: Levenshtein, Jaccard, WordNet, TF-IDF, Word2Vec, BERT, LLM2Vec, Gemini Embeddings
- Three threshold optimization strategies (F1-Gap, Diagonal GT, True GT)
- Further dimension approaches: Tuple Matching, Best-Of-Tuple Matching, Consensus Matching
- Optional preprocessing: Lemmatization, Relevant Clause Extraction (RCE)
- Heatmap visualization of sentence-task similarity matrices
- REST API via Flask as an Interface for the Integration in external applications



## Installation

### 1. Clone the repository

```
git clone https://github.com/Ilya-Az/Model2Text-Comparison---Towards-Accurate-and-Fully-Automated-Similarity-Computation.git
```

### 2. Install dependencies

```
pip install numpy scikit-learn matplotlib seaborn pandas
pip install rapidfuzz nltk gensim spacy flask
pip install sentence-transformers
pip install torch transformers peft==0.11.1 llm2vec
pip install google-genai python-dotenv
```

### 3. Download the spaCy language model

```
python3 -m spacy download en_core_web_sm
```

### 4. Download required NLTK packages

Run the following once in a Python Terminal or Script before first use:

```
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 5. Set up your Gemini API key

Create a .env file in the root directory:

```
GEMINI_API_KEY=your_api_key
```

The .env file is listed in .gitignore and will not be pushed to GitHub.


## Usage

### Run the API server

```
python3 API.py
```

The Flask server starts locally. Send a POST request with the following JSON body:

```
{
  "similarity_panel": true,
  "text": "Your process description",
  "bpmn_xml": "Your BPMN/CPEE XML",
  "approach": "Insert Further Dimension Approach",
  "methods": [
    "method1",
    "method2"
  ],
  "lemmatize": Boolean,
  "remove_cond": Boolean
}
```


## Text Similarity Methods

### Available Methods (JSON input)

```
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
```

### Methods by Type

| Method | Type |
|---|---|
| Levenshtein | Syntactic |
| Jaccard | Syntactic |
| WordNet | Knowledge-based |
| TF-IDF | Corpus-based |
| Word2Vec | Corpus-based |
| BERT (stsb-roberta-large) | Contextual Embedding |
| LLM2Vec (Sheared-LLaMA) | Contextual Embedding |
| Gemini Embeddings (gemini-embedding-2) | Contextual Embedding |


## Further Dimension Approaches

```
APPROACHES = ["tuple", "best_of_tuple", "consensus"]
```


## Project Structure

```
├── Datasets Folder   # stores the Datasets of the PMo Dataset which is used in this work
├── Embedding Folder   # stores the precomputed embeddings for the Datasets
├── Datasets.py     # Task & sentence extraction, lemmatization, RCE, GT data
├── Background.py      # Traditional similarity methods & Match-based F1
├── New_And_State_Of_The_Art.py  # BERT, LLM2Vec, Gemini Embeddings
├── Threshold_Strategies.py   # Three threshold optimization strategies
├── Further_Dimension.py   # Tuple Matching, Best-Of-Tuple, Consensus Matching
├── Autobpmn_ai_service.py   # Facade layer for AutoBPMN.AI
├── API.py        # Flask REST API
├── Evaluation.py      # Spearman correlation, Jaccard Index, GT-F1 evaluation
└── .env          # Gemini API key (not tracked by git)
```