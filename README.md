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

```bash
pip install numpy scikit-learn matplotlib seaborn pandas
pip install rapidfuzz nltk gensim spacy flask
pip install sentence-transformers
pip install torch transformers peft==0.11.1 llm2vec
pip install google-genai python-dotenv
```

### 3. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 4. Download required NLTK packages

Uncomment the following lines in the Background.py class in line 57 to 59 before first use and run the class:

```
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
```
Comment it afterwards again or remove it. 

### 5. Set up your Gemini API key

Create an environment file in the root directory with the name: .env

```
GEMINI_API_KEY=your_api_key
```

The .env file is listed in .gitignore and will not be pushed to GitHub.


## Usage

### Run the API server

```bash
python API.py
```

The Flask server starts locally. Send a POST request with the following JSON body (e.g. via Postman).
Replace the placeholders with the appropriate input:
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

One example for MacOS using the terminal:
```bash
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
  "similarity_panel": true,
  "text": "The customer places an order. We receive the order and process the payment. Finally, the goods are shipped to the customer.",
  "bpmn_xml": "<testset xmlns=\"http://cpee.org/ns/properties/2.0\"><description><description xmlns=\"http://cpee.org/ns/description/1.0\"><call id=\"a1\" endpoint=\"auto\"><parameters><label>Receive customer order</label></parameters></call><call id=\"a2\" endpoint=\"auto\"><parameters><label>Check inventory</label></parameters></call><call id=\"a3\" endpoint=\"auto\"><parameters><label>Process payment</label></parameters></call><call id=\"a4\" endpoint=\"auto\"><parameters><label>Ship goods</label></parameters></call></description></description></testset>",
  "approach": "model2text",
  "methods": [
    {"traditional": "levenshtein"}
  ],
  "lemmatize": false,
  "remove_cond": true
}'
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