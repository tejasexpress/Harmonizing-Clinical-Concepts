# enhanced_transformer_hybrid_search.py

# ----------------------------------------------------------------------
# Environment Setup
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz


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
snomed_df.dropna(subset=['STR', 'CODE'], inplace=True)
rxnorm_df.dropna(subset=['STR', 'CODE'], inplace=True)
snomed_df['CODE'] = snomed_df['CODE'].astype(str)
rxnorm_df['CODE'] = rxnorm_df['CODE'].astype(str)

print(f"Step 3: Data loaded. SNOMED: {len(snomed_df):,}, RxNorm: {len(rxnorm_df):,}, Test: {len(test_df):,}")

# ----------------------------------------------------------------------
# Transformer Model Loading
# ----------------------------------------------------------------------
print("Loading SentenceTransformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Step 4: Transformer model loaded successfully.")

# ----------------------------------------------------------------------
# Embedding Generation and Caching
# ----------------------------------------------------------------------
def encode_in_batches(texts, batch_size=512):
    """
    Encodes text data in batches to manage memory efficiently.
    Returns a stacked numpy array of embeddings.
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        emb = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
        embeddings.append(emb)
    return np.vstack(embeddings)

# Check if embeddings are cached, otherwise compute and save
snomed_embedding_path = "/content/drive/MyDrive/snomed_embeddings.npy"
rxnorm_embedding_path = "/content/drive/MyDrive/rxnorm_embeddings.npy"


print("Computing embeddings for SNOMED...")
snomed_embeddings = encode_in_batches(snomed_df["STR"])

print("\nComputing embeddings for RxNorm...")
rxnorm_embeddings = encode_in_batches(rxnorm_df["STR"])

# Save for future runs
print("\nSaving embeddings to Google Drive for future use...")
np.save(snomed_embedding_path, snomed_embeddings)
np.save(rxnorm_embedding_path, rxnorm_embeddings)
print("Embeddings computed and saved successfully.")

print("Step 5: Embeddings are ready.")

# ----------------------------------------------------------------------
# FAISS Index Setup
# ----------------------------------------------------------------------
print("Building FAISS indexes...")
dim = snomed_embeddings.shape[1]

# Create and populate SNOMED index
index_snomed = faiss.IndexFlatIP(dim)
faiss.normalize_L2(snomed_embeddings)
index_snomed.add(snomed_embeddings)

# Create and populate RxNorm index
index_rxnorm = faiss.IndexFlatIP(dim)
faiss.normalize_L2(rxnorm_embeddings)
index_rxnorm.add(rxnorm_embeddings)

print(f"FAISS indexes built. SNOMED: {index_snomed.ntotal} vectors, RxNorm: {index_rxnorm.ntotal} vectors")
print("Step 6: FAISS semantic indexes are ready.")

# ----------------------------------------------------------------------
# Enhanced Matcher Class Definition
# ----------------------------------------------------------------------
class EnhancedMatcher:
    """
    Handles entity-aware preprocessing, multi-factor scoring, and intelligent fallback
    for clinical concept harmonization.
    """
    
    def __init__(self):
        # Medical abbreviations and normalizations
        self.medical_abbrev = {
            # Lab abbreviations
            'hb': 'haemoglobin', 'hgb': 'haemoglobin', 'hemoglobin': 'haemoglobin',
            'rbc': 'red blood cell', 'wbc': 'white blood cell',
            'esr': 'erythrocyte sedimentation rate', 'crp': 'c-reactive protein',
            'ldl': 'low density lipoprotein', 'hdl': 'high density lipoprotein',
            'tsh': 'thyroid stimulating hormone', 'ft4': 'free thyroxine',
            'alt': 'alanine transaminase', 'ast': 'aspartate transaminase',
            'bun': 'blood urea nitrogen', 'egfr': 'estimated glomerular filtration rate',
            
            # Procedure abbreviations
            'xr': 'x-ray', 'ct': 'computed tomography', 'mri': 'magnetic resonance imaging',
            'ecg': 'electrocardiogram', 'ekg': 'electrocardiogram',
            'echo': 'echocardiogram', 'us': 'ultrasound', 'usg': 'ultrasound',
            
            # Medical terms
            'bp': 'blood pressure', 'hr': 'heart rate', 'temp': 'temperature',
            'resp': 'respiratory', 'abd': 'abdominal', 'gi': 'gastrointestinal',
            
            # Drug abbreviations
            'tab': 'tablet', 'cap': 'capsule', 'inj': 'injection',
            'mg': 'milligram', 'ml': 'milliliter', 'mcg': 'microgram',
            'iu': 'international unit'
        }
        
        # Common drug name mappings
        self.drug_synonyms = {
            'paracetamol': 'acetaminophen',
            'panadol': 'acetaminophen',
            'tylenol': 'acetaminophen',
            'crocin': 'acetaminophen',
            'aspro': 'aspirin',
            'disprin': 'aspirin',
            'combiflam': 'ibuprofen',
            'brufen': 'ibuprofen',
            'calpol': 'acetaminophen'
        }
        
        # TTY priority for RxNorm (higher = better)
        self.rxnorm_tty_priority = {
            'SCD': 10, 'SBD': 9, 'GPCK': 8, 'BPCK': 7, 'SCDC': 6,
            'SBDC': 5, 'BN': 4, 'IN': 3, 'PIN': 2, 'MIN': 1
        }
        
        # TTY priority for SNOMED
        self.snomed_tty_priority = {
            'PT': 10, 'SY': 5, 'FN': 3
        }
    
    def preprocess_text(self, text, entity_type):
        """
        Applies entity-aware preprocessing including abbreviation expansion,
        drug synonym mapping, and anatomical standardization.
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Entity-specific preprocessing
        if entity_type.lower() == 'medication':
            # Handle drug name synonyms
            for synonym, standard in self.drug_synonyms.items():
                if synonym in text:
                    text = text.replace(synonym, standard)
            
            # Standardize dosage formats
            text = re.sub(r'(\d+)\s*mg', r'\1 milligram', text)
            text = re.sub(r'(\d+)\s*ml', r'\1 milliliter', text)
        
        elif entity_type.lower() == 'diagnosis':
            # Handle anatomical locations
            location_map = {
                'upper right abdomen': 'right upper quadrant abdominal',
                'upper left abdomen': 'left upper quadrant abdominal',
                'stomach': 'gastric', 'belly': 'abdominal'
            }
            for informal, formal in location_map.items():
                if informal in text:
                    text = text.replace(informal, formal)
        
        elif entity_type.lower() == 'procedure':
            # Handle surgical procedures
            surgery_map = {
                'appendix removal': 'appendectomy',
                'gallbladder removal': 'cholecystectomy',
                'removal surgery': 'removal',
                'surgery': ''
            }
            for layman, medical in surgery_map.items():
                if layman in text:
                    text = text.replace(layman, medical)
        
        elif entity_type.lower() == 'lab':
            # Standardize lab test names
            lab_map = {
                'blood sugar': 'glucose',
                'sugar test': 'glucose measurement',
                'fasting sugar': 'glucose measurement fasting',
                'blood count': 'complete blood count'
            }
            for informal, formal in lab_map.items():
                if informal in text:
                    text = text.replace(informal, formal)
        
        # General abbreviation expansion
        for abbrev, full in self.medical_abbrev.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def get_tty_priority_score(self, tty, system):
        """Returns normalized priority score for term type."""
        if system == 'RXNORM':
            return self.rxnorm_tty_priority.get(tty, 0)
        else:
            return self.snomed_tty_priority.get(tty, 0)
    
    def calculate_enhanced_score(self, query, candidate_row, semantic_score, system):
        """
        Calculates multi-factor score combining semantic similarity, lexical similarity,
        TTY priority, and length similarity.
        """
        candidate_text = candidate_row['STR']
        
        # Multiple lexical similarity scores
        token_set_score = fuzz.token_set_ratio(query, candidate_text) / 100
        token_sort_score = fuzz.token_sort_ratio(query, candidate_text) / 100
        partial_score = fuzz.partial_ratio(query, candidate_text) / 100
        
        # Take the best lexical score
        lexical_score = max(token_set_score, token_sort_score, partial_score)
        
        # TTY priority score (normalized to 0-1)
        tty_score = self.get_tty_priority_score(candidate_row['TTY'], system) / 10
        
        # Length similarity bonus
        len_diff = abs(len(query.split()) - len(candidate_text.split()))
        len_bonus = max(0, 1 - len_diff * 0.05)
        
        # Final weighted score
        final_score = (0.35 * semantic_score + 
                      0.40 * lexical_score + 
                      0.15 * tty_score + 
                      0.10 * len_bonus)
        
        return final_score
    
    def fallback_search(self, query, entity_type, df, system):
        """
        Intelligent fallback strategy for low-confidence matches.
        For medications, attempts ingredient-based matching.
        """
        # For medications, try to find base ingredient
        if entity_type.lower() == 'medication' and system == 'RXNORM':
            key_terms = re.findall(r'\b[a-zA-Z]{4,}\b', query)
            for term in key_terms:
                matches = df[df['STR'].str.contains(term, case=False, na=False)]
                ingredient_matches = matches[matches['TTY'] == 'IN']
                if not ingredient_matches.empty:
                    best_match = ingredient_matches.iloc[0]
                    return system, str(best_match['CODE']), best_match['STR']
        
        # Generic fallback - find best fuzzy match
        best_score, best_row = -1, None
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for _, row in sample_df.iterrows():
            score = fuzz.token_set_ratio(query, row['STR']) / 100
            if score > best_score:
                best_score, best_row = score, row
        
        if best_row is not None and best_score > 0.3:
            return system, str(best_row['CODE']), best_row['STR']
        
        return system, "NOT_FOUND", "No suitable match found"

print("Step 7: Enhanced matcher class is defined.")

# ----------------------------------------------------------------------
# Harmonization Function Definition
# ----------------------------------------------------------------------
# Initialize enhanced matcher
matcher = EnhancedMatcher()

def enhanced_hybrid_search(query, entity_type, top_k=20):
    """
    Performs enhanced hybrid search combining semantic embeddings, lexical matching,
    and entity-aware preprocessing with intelligent fallback strategies.
    """
    # Preprocess query based on entity type
    processed_query = matcher.preprocess_text(query, entity_type)
    query_to_use = processed_query if processed_query else query
    
    # Encode query using transformer model
    query_vec = model.encode([query_to_use])
    faiss.normalize_L2(query_vec)
    
    # Select appropriate terminology system and index
    if entity_type.lower() in ["diagnosis", "procedure", "lab"]:
        df = snomed_df
        index = index_snomed
        system = "SNOMEDCT_US"
    else:
        df = rxnorm_df
        index = index_rxnorm
        system = "RXNORM"
    
    # Retrieve top-k candidates using semantic similarity
    D, I = index.search(query_vec, top_k)
    candidates = df.iloc[I[0]]
    
    # Score all candidates with enhanced multi-factor scoring
    best_score, best_row = -1, None
    for idx, (_, candidate_row) in enumerate(candidates.iterrows()):
        semantic_score = D[0][idx]
        
        final_score = matcher.calculate_enhanced_score(
            query_to_use, candidate_row, semantic_score, system
        )
        
        if final_score > best_score:
            best_score, best_row = final_score, candidate_row
    
    # Apply fallback strategy if confidence is below threshold
    if best_score < 0.5:
        return matcher.fallback_search(query_to_use, entity_type, df, system)
    
    return system, str(best_row["CODE"]), best_row["STR"]

print("Step 8: Enhanced harmonization function is defined.")

# ----------------------------------------------------------------------
# Batch Processing
# ----------------------------------------------------------------------
results = []

print("\nProcessing test data with enhanced hybrid search...")
for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Harmonizing Entities"):
    input_text = str(row["Input Entity Description"])
    entity_type = str(row["Entity Type"])
    
    try:
        system, code, desc = enhanced_hybrid_search(input_text, entity_type)
        
        results.append({
            'Output Coding System': system,
            'Output Target Code': code,
            'Output Target Description': desc
        })
        
    except Exception as e:
        print(f"\nError processing row {index}: {e}")
        results.append({
            'Output Coding System': 'ERROR',
            'Output Target Code': 'ERROR',
            'Output Target Description': str(e)
        })

print("\nStep 9: Batch harmonization process is complete.")

# ----------------------------------------------------------------------
# Save Results
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results)
final_df = pd.concat([test_df, results_df], axis=1)
output_filename = 'Test_Enhanced_Predictions.xlsx'
final_df.to_excel(output_filename, index=False)

print(f"\nAll steps completed successfully. Results saved to '{output_filename}'")

# Display sample results
print("\nSample Results:")
display_cols = ["Input Entity Description", "Entity Type", "Output Coding System", 
                "Output Target Code", "Output Target Description"]
print(final_df[display_cols].head(10).to_string(index=False))