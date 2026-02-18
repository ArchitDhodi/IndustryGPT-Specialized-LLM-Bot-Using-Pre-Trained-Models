![Developer Docs Assistant Banner](https://images.unsplash.com/photo-1516116216624-53e697fedbea?q=80&w=1600&auto=format&fit=crop)

# Industry-Specific LLM Bot - Technology & IT (Developer Docs Assistant)

**End-to-End NLP and Generative AI System for Developer Documentation Q&A**  
Built with document ingestion, preprocessing, retrieval-augmented generation (RAG), local fine-tuning, evaluation, and Streamlit deployment.

---

## Project Overview

This project implements an industry-focused LLM bot for the **Technology & IT** domain.
The assistant is trained and tuned to answer practical developer questions using technical documentation from:

- NumPy
- pandas
- scikit-learn
- Docker
- Kubernetes
- Airflow
- MLflow

The system is designed as a complete pipeline, not a notebook demo:

- Data collection from official documentation sources
- Text cleaning and chunking
- Baseline retrieval + FAISS vector search
- RAG question-answering with source attribution
- LoRA-based fine-tuning on curated QA pairs
- Post-training comparison (base vs fine-tuned)
- Interactive app deployment with Streamlit

---

## Problem Statement

Developer teams spend significant time searching across fragmented documentation.
This project solves that by building an intelligent assistant that can:

- Understand domain-specific technical queries
- Retrieve relevant documentation context
- Generate structured, practical answers with examples
- Improve response quality through fine-tuning

---

## Objective Coverage (Capstone Requirements)

### 1) Industry Selection
- Selected industry: **Technology & Information Technology (IT)**
- Focus area: **Developer documentation assistant**

### 2) Data Collection
- Collected multi-source documentation text from official docs
- Stored raw and cleaned corpora in project data folders
- Built a curated QA dataset for fine-tuning

### 3) Model Selection and Training
- Base LLM: **google/flan-t5-base** (Hugging Face)
- Embeddings + retrieval stack: **Sentence Transformers + FAISS + LangChain**
- Fine-tuning: **LoRA-based local training** with saved model artifacts
- Training setup supports capstone constraint (up to 25 epochs)

### 4) Bot Development
- Built a working docs assistant with:
  - Query routing by library/topic
  - Retrieval-augmented generation
  - Source links in output
  - Structured answer + example format

### 5) Demonstration Readiness
- Working Streamlit application (`dashboard.py`)
- Notebook pipeline (`bot.ipynb`) with evaluation exports
- Comparison artifacts for base vs fine-tuned behavior

### 6) Research Paper Readiness
- Includes cleaned datasets, model pipeline, evaluation artifacts, and limitations/future-work notes suitable for narrative synthesis in research writing.

---

## System Architecture

Raw Docs -> Cleaning -> Chunking -> Baseline Retrieval -> FAISS Index -> RAG Inference -> QA Dataset Creation -> LoRA Fine-Tuning -> Base vs Fine-Tuned Evaluation -> Streamlit App

---

## Data Engineering Pipeline

### Data Collection and Storage
- Scraped and parsed official documentation pages
- Saved raw corpus: `data/raw/docs_raw.csv`

### Cleaning and Preprocessing
- Removed noise and malformed text
- Standardized content for chunking and retrieval
- Saved cleaned corpus: `data/processed/docs_clean.csv`

### Chunking
- Split long documents into overlapping chunks for retrieval quality
- Saved chunked corpus: `data/processed/doc_chunks.csv`

---

## Retrieval and Generation Stack

### Baseline Retrieval
- TF-IDF baseline for quick retrieval sanity checks

### RAG Layer
- FAISS vector index from cleaned chunks
- LangChain RetrievalQA chain
- Prompt-constrained answer format with source grounding

### Output Format
- One direct answer sentence
- One short practical Python example
- Source URLs from retrieved docs

---

## Fine-Tuning Pipeline

- QA pair generation from curated docs
- Train/validation split JSONL creation
- LoRA fine-tuning on `flan-t5-base`
- Saved model artifacts:
  - `finetuned_docs_bot/`
  - `finetuned_docs_bot_adapter/`

Evaluation artifacts:
- `artifacts/eval_results.csv`
- `artifacts/finetune_eval_comparison.csv`

---

## Interactive Deployment (Streamlit)

Application file: `dashboard.py`

Features:
- Ask technical documentation questions
- Automatic topic-aware query routing
- Structured answer + code example rendering
- Source citation display
- Uses local FAISS index and local fine-tuned model path

Run command:

```bash
streamlit run dashboard.py
```

---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Hugging Face Transformers
- PEFT (LoRA)
- Sentence Transformers
- FAISS
- LangChain
- Streamlit

---

## Project Structure

```text
ABSM Project 6/
|-- bot.ipynb
|-- dashboard.py
|-- data/
|   |-- raw/docs_raw.csv
|   `-- processed/
|       |-- docs_clean.csv
|       |-- doc_chunks.csv
|       |-- qa_seed.jsonl
|       |-- qa_train.jsonl
|       |-- qa_train_split.jsonl
|       `-- qa_val_split.jsonl
|-- artifacts/
|   |-- eval_results.csv
|   |-- finetune_eval_comparison.csv
|   `-- faiss_index/
|-- finetuned_docs_bot/
|-- finetuned_docs_bot_adapter/
`-- README.md
```

---

## Setup Instructions

```bash
# 1) Create and activate environment
conda create -n docs-bot python=3.10 -y
conda activate docs-bot

# 2) Install dependencies (example)
pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib seaborn
pip install transformers datasets peft accelerate sentence-transformers
pip install langchain langchain-community langchain-text-splitters faiss-cpu streamlit

# 3) Run notebook pipeline
# Open bot.ipynb and run sections in order

# 4) Launch app
streamlit run dashboard.py
```

---

## Business and Academic Value

- Reduces documentation lookup time for developers and learners
- Improves technical support quality through grounded answers
- Demonstrates full lifecycle GenAI system design for portfolio and interviews
- Provides reproducible artifacts for research-paper extension

---

## Author

**Archit Dhodi**

---

## License

Academic, learning, and portfolio demonstration use.
