# extract_embeddings.py

import os
import json
import random
import pickle
import torch
from sentence_transformers import SentenceTransformer
import sys

# =====================================================================
# CONFIGURATION
# =====================================================================
# Update this section in your script to match your actual folder names on disk
DATASETS = ["trec_covid", "nfcorpus", "webis_touche", "scifact", "fiqa", "scidocs", "arguana", "quora", 
            "physics", "english", "gis", "stats", "mathematica", "webmasters", "wordpress", "programmers",
            "unix", "android", "gaming", "tex"] 

# MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_NAME = "intfloat/e5-large-v2"


if MODEL_NAME == "BAAI/bge-large-en-v1.5":
    BASE_EXPORT_DIR = "./exports_bge"
elif MODEL_NAME == "intfloat/e5-large-v2":
    BASE_EXPORT_DIR = "./exports_e5"
else:
    sys.exit(f"Error: Unsupported MODEL_NAME '{MODEL_NAME}'. Stopping script.")



TOP_K = 1000            # Top-K candidate set size per query
NUM_QUERIES = None      # Number of random queries to sample from dataset
SEED = 111
BATCH_SIZE = 32         # depending upon the VRAM or laptop configurations
datasets_path = "datasets"  # beir downloaded path

# =====================================================================
# DATASET LOADER FUNCTION
# =====================================================================
def load_local_dataset(dataset_name: str) -> tuple:
    data_path = f"./{datasets_path}/{dataset_name}"
    corpus_file = os.path.join(data_path, "corpus.jsonl")
    queries_file = os.path.join(data_path, "queries.jsonl")
    
    if not os.path.exists(corpus_file) or not os.path.exists(queries_file):
        raise FileNotFoundError(f"Could not find dataset files at '{data_path}'. "
                                f"Ensure corpus.jsonl and queries.jsonl exist.")
        
    documents, doc_ids = [], []
    queries, query_ids = [], []
    
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            title = data.get("title", "")
            text = data.get("text", data.get("txt", ""))
            
            if MODEL_NAME == "BAAI/bge-large-en-v1.5":
                full_text = f"{title} {text}".strip()
            elif MODEL_NAME == "intfloat/e5-large-v2":
                full_text = f"passage: {title} {text}".strip()
            else:
                raise ValueError(f"Unsupported model: {MODEL_NAME}")

            doc_id = data.get("_id", data.get("id", f"doc_{idx}"))
            if full_text:
                documents.append(full_text)
                doc_ids.append(str(doc_id))
                
    print(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            q_text = data.get("text", "")
            q_id = data.get("_id", data.get("id", f"q_{idx}"))
            if q_text:
                queries.append(q_text)
                query_ids.append(str(q_id))
                
    return (doc_ids, documents), (query_ids, queries)


# =====================================================================
# MAIN ENTRY POINT & MULTI-DATASET LOOP
# =====================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Shared Transformer Model on [{device.upper()}]: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    for dataset_name in DATASETS:
        print("\n" + "="*60)
        print(f"    EXTRACTING DISTANCE MATRICES FOR DATASET: {dataset_name.upper()}")
        print("="*60)
        
        # Setup Export Directory
        dataset_export_dir = os.path.join(BASE_EXPORT_DIR, dataset_name)
        os.makedirs(dataset_export_dir, exist_ok=True)
        corpus_pkl_path = os.path.join(dataset_export_dir, "corpus_embeddings.pkl")
        
        # 1. Load Local Dataset
        (doc_ids, documents), (query_ids, queries_list) = load_local_dataset(dataset_name)
        print(f"Loaded {len(documents)} documents and {len(queries_list)} total queries.")
    
        
        # 2. Cache Check / Generate Corpus Embeddings
        if os.path.exists(corpus_pkl_path):
            print(f"\n[CACHE HIT] Found existing corpus embeddings at '{corpus_pkl_path}'.")
            with open(corpus_pkl_path, 'rb') as f:
                cached_data = pickle.load(f)
                doc_embeddings = cached_data["doc_embeddings"]
            print(f"Loaded cached embeddings matrix: {doc_embeddings.shape}")
        else:
            print(f"\n[CACHE MISS] Encoding Corpus Vectors for '{dataset_name}'...")
            doc_embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
            
            print(f"Saving Corpus Embeddings to: {corpus_pkl_path}")
            with open(corpus_pkl_path, 'wb') as f:
                pickle.dump({"doc_ids": doc_ids, "doc_embeddings": doc_embeddings}, f)