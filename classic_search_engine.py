# classic_search_engine.py

# ----------------------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import nltk
import re
import os
from tqdm import tqdm  # Changed from tqdm.notebook for script compatibility

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("Step 1: Environment setup is complete.")

# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
# Load the test data
test_df = pd.read_excel('Test.xlsx')

# Load the target terminology files
snomed_df = pd.read_parquet('snomed_all_data.parquet')
rxnorm_df = pd.read_parquet('rxnorm_all_data.parquet')

# Clean the data
snomed_df.dropna(subset=['STR', 'CUI', 'CODE'], inplace=True)
rxnorm_df.dropna(subset=['STR', 'CUI', 'CODE'], inplace=True)
snomed_df['CODE'] = snomed_df['CODE'].astype(str)
rxnorm_df['CODE'] = rxnorm_df['CODE'].astype(str)

print(f"Step 2: Data loaded. SNOMED DF size: {len(snomed_df):,}, RxNorm DF size: {len(rxnorm_df):,}")

# ----------------------------------------------------------------------
# Preprocessing Function
# ----------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
# Define custom stopwords, keeping relevant medical abbreviations
custom_stopwords = set(stopwords.words('english')) - {'mg', 'ml', 'tablet', 'tablets'}

def preprocess_text(text):
    """
    Cleans and tokenizes text for search indexing.
    - Converts to lowercase
    - Removes punctuation
    - Lemmatizes words
    - Removes stopwords
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in custom_stopwords]
    return tokens

print("Step 3: Preprocessing function is defined.")

# ----------------------------------------------------------------------
# BM25 Lexical Indexing
# ----------------------------------------------------------------------
# Preprocess the STR columns for BM25
print("Tokenizing SNOMED corpus for BM25 indexing...")
snomed_corpus = [preprocess_text(text) for text in tqdm(snomed_df['STR'])]

print("\nTokenizing RxNorm corpus for BM25 indexing...")
rxnorm_corpus = [preprocess_text(text) for text in tqdm(rxnorm_df['STR'])]

# Create BM25 indexes
print("\nIndexing SNOMED for BM25...")
bm25_snomed = BM25Okapi(snomed_corpus)

print("Indexing RxNorm for BM25...")
bm25_rxnorm = BM25Okapi(rxnorm_corpus)

print("\n BM25 lexical indexes are ready.")

# ----------------------------------------------------------------------
# FAISS Semantic Indexing
# ----------------------------------------------------------------------
# Load the sentence transformer model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_and_save_faiss_index(df, column_name, model, file_prefix):
    """Encodes text data, creates a FAISS index, and saves it to disk to avoid re-computing."""
    index_file = f'{file_prefix}_faiss.index'

    if os.path.exists(index_file):
        print(f"Loading pre-computed FAISS index from {index_file}")
        faiss_index = faiss.read_index(index_file)
    else:
        print(f"Computing embeddings for {file_prefix}... (This will take a while the first time!)")
        embeddings = model.encode(df[column_name].tolist(), show_progress_bar=True, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        print(f"Building and saving FAISS index to {index_file}...")
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        faiss.write_index(faiss_index, index_file)
        
    print(f"Index for {file_prefix} is ready. Contains {faiss_index.ntotal} vectors.")
    return faiss_index

# Create and save the indexes for both terminologies
print("\n--- Setting up SNOMED Semantic Search ---")
faiss_snomed = create_and_save_faiss_index(snomed_df, 'STR', semantic_model, 'snomed')

print("\n--- Setting up RxNorm Semantic Search ---")
faiss_rxnorm = create_and_save_faiss_index(rxnorm_df, 'STR', semantic_model, 'rxnorm')

print("\n Step 4: All search indexes are ready.")

# ----------------------------------------------------------------------
# Cross-Encoder Re-ranker Setup
# ----------------------------------------------------------------------
print("Loading Cross-Encoder re-ranking model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Step 5: Cross-Encoder is ready.")

# ----------------------------------------------------------------------
# Harmonization Function Definition
# ----------------------------------------------------------------------
def harmonize_entity_upgraded(description, entity_type, top_k=75):
    """
    Finds the best match using a robust hybrid search and cross-encoder re-ranking.
    """
    # 1. Select the correct tools based on entity type
    if entity_type in ['Procedure', 'Lab', 'Diagnosis']:
        target_df, bm25_index, faiss_index = snomed_df, bm25_snomed, faiss_snomed
    elif entity_type == 'Medicine':
        target_df, bm25_index, faiss_index = rxnorm_df, bm25_rxnorm, faiss_rxnorm
    else:
        return "UNKNOWN_ENTITY", "N/A", "N/A", "N/A"

    # 2. Preprocess the query for BM25
    query_tokens = preprocess_text(description)
    if not query_tokens:
        return "NO_MATCH_FOUND", "N/A", "N/A", "N/A"

    # --- STAGE 1: HYBRID CANDIDATE RETRIEVAL ---
    # a) Lexical search (BM25)
    lexical_indices = bm25_index.get_top_n(query_tokens, target_df.index, n=top_k)
    
    # b) Semantic search (FAISS)
    query_embedding = semantic_model.encode(description)
    _, semantic_indices_loc = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    semantic_indices = target_df.index[semantic_indices_loc[0]].tolist()
    
    # c) Merge candidates and retrieve unique descriptions
    combined_indices = list(set(lexical_indices) | set(semantic_indices))
    if not combined_indices:
        return "NO_MATCH_FOUND", "N/A", "N/A", "N/A"
    candidate_df = target_df.loc[combined_indices]

    # --- STAGE 2: CROSS-ENCODER RE-RANKING ---
    # Create [query, candidate] pairs for the model
    rerank_pairs = [[description, str(candidate_str)] for candidate_str in candidate_df['STR']]
    
    # Get highly accurate relevance scores
    scores = cross_encoder.predict(rerank_pairs)
    
    # Find the best match
    best_candidate_local_index = np.argmax(scores)
    best_match_original_index = candidate_df.index[best_candidate_local_index]
    best_match = target_df.loc[best_match_original_index]
    
    return best_match['System'], best_match['CODE'], best_match['STR'], best_match['CUI']

print("Step 6: Final upgraded harmonization function is defined.")

# ----------------------------------------------------------------------
# Batch Processing
# ----------------------------------------------------------------------
results = []

for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Harmonizing Entities"):
    description = row['Input Entity Description']
    entity_type = row['Entity Type']
    
    # Call the harmonization function
    system, code, standard_desc, cui = harmonize_entity_upgraded(description, entity_type)
    
    results.append({
        'Output Coding System': system,
        'Output Target Code': code,
        'Output Target Description': standard_desc,
        'Output Target CUI': cui
    })

print("\nStep 7: Batch harmonization process is complete.")

# ----------------------------------------------------------------------
# Save Results
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results)
final_df = pd.concat([test_df, results_df], axis=1)
output_filename = 'Test_predictions.xlsx'
final_df.to_excel(output_filename, index=False)

print(f"All steps completed. Results saved to '{output_filename}'")