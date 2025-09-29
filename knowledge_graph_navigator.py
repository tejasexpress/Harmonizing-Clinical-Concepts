"""
Knowledge Graph Navigator for Harmonizing Clinical Data

This module implements a graph-based approach to harmonize clinical concepts
by mapping input clinical descriptions to standardized codes from RxNorm or SNOMED CT.
"""

import pandas as pd
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import yaml
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalDataPreprocessor:
    """Handles preprocessing of clinical text data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'these',
            'they', 'their', 'there', 'have', 'had', 'do', 'does', 'did'
        ])
        
        # Load medical abbreviations from config
        self.medical_abbreviations = config.get('medical_abbreviations', {})
    
    def clean_text(self, text: str) -> Tuple[str, bool, str]:
        """Clean and normalize text, return (cleaned_text, abbreviations_expanded, original_before_expansion)"""
        if pd.isna(text) or not isinstance(text, str):
            return "", False, ""
        
        # Convert to lowercase
        original_text = text.lower().strip()
        text = original_text
        abbreviations_expanded = False
        
        # Preserve decimal numbers and percentages by temporarily replacing them
        # Find all decimal numbers with units (e.g., "0.1 mg", "0.01%", "2.5 ml")
        decimal_patterns = []
        
        # Pattern for decimal numbers with optional units (including percentages)
        decimal_regex = r'(\d+\.?\d*)\s*(%|mg/ml|mg/gram|unit/ml|mg|ml|gram|unit|mcg|kg|lb|oz)\b'
        matches = list(re.finditer(decimal_regex, text))
        
        # Store original matches and replace with placeholders
        for i, match in enumerate(matches):
            placeholder = f"__DECIMAL_{i}__"
            # Normalize the unit representation
            normalized = match.group(0).replace(' %', '%').strip()
            decimal_patterns.append((placeholder, normalized))
            text = text.replace(match.group(0), placeholder, 1)
        
        # Also preserve standalone decimal numbers and percentages
        standalone_decimal_regex = r'\b\d+\.\d+%?\b'
        standalone_matches = list(re.finditer(standalone_decimal_regex, text))
        
        for i, match in enumerate(standalone_matches):
            if match.group(0) not in [dp[1] for dp in decimal_patterns]:  # Avoid duplicates
                placeholder = f"__STANDALONE_{i}__"
                decimal_patterns.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder, 1)
        
        # Remove parentheses but keep their content (e.g., "(0.1 mg/gram)" -> "0.1 mg/gram")
        text = re.sub(r'[()]', ' ', text)
        
        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, text):
                text = re.sub(pattern, expansion, text)
                abbreviations_expanded = True
        
        # Remove most special characters but keep spaces, hyphens, and our placeholders
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Restore decimal numbers and units
        for placeholder, original in decimal_patterns:
            text = text.replace(placeholder, original)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text, abbreviations_expanded, original_text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing stop words"""
        if not text:
            return []
        
        tokens = text.split()
        # Remove stop words and single character tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) >= 1]
        
        return tokens
    
    def preprocess_descriptions(self, descriptions: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Preprocess a list of clinical descriptions and return expansions info"""
        processed = []
        expansions_found = []
        
        for i, desc in enumerate(descriptions):
            cleaned, abbreviations_expanded, original_before_expansion = self.clean_text(desc)
            tokens = self.tokenize(cleaned)
            
            processed.append({
                'original': desc,
                'cleaned': cleaned,
                'tokens': tokens,
                'token_string': ' '.join(tokens),
                'abbreviations_expanded': abbreviations_expanded
            })
            
            if abbreviations_expanded:
                expansions_found.append({
                    'index': i,
                    'original': desc,
                    'before_expansion': original_before_expansion,
                    'after_expansion': cleaned
                })
        
        return processed, expansions_found


class KnowledgeGraphNavigator:
    """Main class for the Knowledge Graph Navigator"""
    
    def __init__(self, config_path: str = 'scoring_config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = ClinicalDataPreprocessor(self.config)
        self.rxnorm_data = None
        self.snomed_data = None
        self.test_data = None
        self.combined_data = None
        self.knowledge_graph = None
        self.bm25_model = None
        self.processed_targets = {}
        
        # Cache for data-driven discoveries
        self._entity_sty_cache = {}
        self._tty_relationships_cache = {}
        
    def load_data(self, rxnorm_path: str = 'Target Description Files/rxnorm_all_data.parquet',
                  snomed_path: str = 'Target Description Files/snomed_all_data.parquet',
                  test_path: str = 'Test.xlsx'):
        """Load SNOMED, RxNorm, and test data"""
        
        logger.info("Loading clinical data...")
        
        # Load SNOMED and RxNorm Data
        self.rxnorm_data = pd.read_parquet(rxnorm_path)
        self.snomed_data = pd.read_parquet(snomed_path)
        
        # Load the test dataset
        self.test_data = pd.read_excel(test_path)
        
        logger.info(f"Loaded RxNorm data: {self.rxnorm_data.shape}")
        logger.info(f"Loaded SNOMED data: {self.snomed_data.shape}")
        logger.info(f"Loaded test data: {self.test_data.shape}")
        
        # Display sample data
        print("\n=== Sample RxNorm Data ===")
        print(self.rxnorm_data.head(3))
        print(f"Unique CUIs in RxNorm: {self.rxnorm_data['CUI'].nunique()}")
        
        print("\n=== Sample SNOMED Data ===")
        print(self.snomed_data.head(3))
        print(f"Unique CUIs in SNOMED: {self.snomed_data['CUI'].nunique()}")
        
        print("\n=== Sample Test Data ===")
        print(self.test_data.head(3))
        print(f"Entity Types in test data: {self.test_data['Entity Type'].value_counts()}")
        
    def preprocess_clinical_data(self):
        """Preprocess the clinical descriptions in the test data"""
        
        logger.info("Preprocessing clinical descriptions...")
        
        # Preprocess test data descriptions
        test_descriptions = self.test_data['Input Entity Description'].tolist()
        self.processed_test, test_expansions = self.preprocessor.preprocess_descriptions(test_descriptions)
        
        # Add processed data to test dataframe
        self.test_data['cleaned_description'] = [item['cleaned'] for item in self.processed_test]
        self.test_data['tokens'] = [item['tokens'] for item in self.processed_test]
        self.test_data['token_string'] = [item['token_string'] for item in self.processed_test]
        
        print("\n=== Preprocessed Test Data Sample ===")
        for i in range(3):
            print(f"Original: {self.processed_test[i]['original']}")
            print(f"Cleaned: {self.processed_test[i]['cleaned']}")
            print(f"Tokens: {self.processed_test[i]['tokens']}")
            print("---")
        
        # Display abbreviation expansions in test data
        if test_expansions:
            print(f"\n=== Abbreviation Expansions Found in Test Data ({len(test_expansions)} items) ===")
            for expansion in test_expansions:
                print(f"Row {expansion['index']}:")
                print(f"  Original: '{expansion['original']}'")
                print(f"  Before expansion: '{expansion['before_expansion']}'")
                print(f"  After expansion: '{expansion['after_expansion']}'")
                print("---")
        else:
            print("\n=== No abbreviation expansions found in test data ===")
        
        # Preprocess target data (RxNorm and SNOMED)
        logger.info("Preprocessing target descriptions...")
        
        # Process RxNorm descriptions
        rxnorm_descriptions = self.rxnorm_data['STR'].tolist()
        self.processed_rxnorm, rxnorm_expansions = self.preprocessor.preprocess_descriptions(rxnorm_descriptions)
        
        # Process SNOMED descriptions  
        snomed_descriptions = self.snomed_data['STR'].tolist()
        self.processed_snomed, snomed_expansions = self.preprocessor.preprocess_descriptions(snomed_descriptions)
        
        # Add processed data to target dataframes
        self.rxnorm_data['cleaned_description'] = [item['cleaned'] for item in self.processed_rxnorm]
        self.rxnorm_data['tokens'] = [item['tokens'] for item in self.processed_rxnorm]
        self.rxnorm_data['token_string'] = [item['token_string'] for item in self.processed_rxnorm]
        
        self.snomed_data['cleaned_description'] = [item['cleaned'] for item in self.processed_snomed]
        self.snomed_data['tokens'] = [item['tokens'] for item in self.processed_snomed]
        self.snomed_data['token_string'] = [item['token_string'] for item in self.processed_snomed]
        
        # Display abbreviation expansions in target data (show first few examples)
        if rxnorm_expansions:
            print(f"\n=== Sample Abbreviation Expansions in RxNorm Data ({len(rxnorm_expansions)} total) ===")
            for expansion in rxnorm_expansions[:5]:  # Show first 5 examples
                print(f"Row {expansion['index']}:")
                print(f"  Original: '{expansion['original']}'")
                print(f"  Before expansion: '{expansion['before_expansion']}'")
                print(f"  After expansion: '{expansion['after_expansion']}'")
                print("---")
        
        if snomed_expansions:
            print(f"\n=== Sample Abbreviation Expansions in SNOMED Data ({len(snomed_expansions)} total) ===")
            for expansion in snomed_expansions[:5]:  # Show first 5 examples
                print(f"Row {expansion['index']}:")
                print(f"  Original: '{expansion['original']}'")
                print(f"  Before expansion: '{expansion['before_expansion']}'")
                print(f"  After expansion: '{expansion['after_expansion']}'")
                print("---")
        
        logger.info("Clinical data preprocessing completed!")
    
    def initialize_bm25(self):
        """Initialize BM25 model with combined RxNorm and SNOMED data"""
        
        logger.info("Initializing BM25 model...")
        
        # Combine RxNorm and SNOMED data with source tracking
        rxnorm_combined = self.rxnorm_data.copy()
        rxnorm_combined['source'] = 'RXNORM'
        rxnorm_combined['source_index'] = range(len(rxnorm_combined))
        
        snomed_combined = self.snomed_data.copy()
        snomed_combined['source'] = 'SNOMEDCT_US'
        snomed_combined['source_index'] = range(len(snomed_combined))
        
        # Combine both datasets
        self.combined_data = pd.concat([rxnorm_combined, snomed_combined], ignore_index=True)
        
        # Create corpus from tokenized descriptions and metadata
        logger.info("Creating BM25 corpus...")
        corpus = []
        
        for idx, row in self.combined_data.iterrows():
            # Combine tokens with system info
            doc_tokens = row['tokens'].copy() if row['tokens'] and len(row['tokens']) > 0 else ['']
            doc_tokens.append(row['System'].lower())  # Add system info
            corpus.append(doc_tokens)
        
        # Initialize BM25 model
        logger.info("Training BM25 model...")
        self.bm25_model = BM25Okapi(corpus)
        
        logger.info(f"BM25 model initialized with {len(corpus)} documents")
        print(f"Combined dataset shape: {self.combined_data.shape}")
        print(f"RxNorm entries: {len(rxnorm_combined)}")
        print(f"SNOMED entries: {len(snomed_combined)}")
    
    def get_bm25_candidates(self, description_tokens: List[str], entity_type: str = None, top_k: int = None) -> List[Dict]:
        """Get top-k BM25 candidates for a given description and entity type"""
        
        if not self.bm25_model:
            raise ValueError("BM25 model not initialized. Call initialize_bm25() first.")
        
        if not description_tokens or len(description_tokens) == 0:
            return []
        
        # Combine description tokens with entity type for matching
        search_tokens = description_tokens.copy()
        if entity_type:
            # Add entity type as part of the search query
            search_tokens.extend(entity_type.lower().split())
        
        # Get BM25 scores
        scores = self.bm25_model.get_scores(search_tokens)
        
        # Get top-k from config if not specified
        if top_k is None:
            top_k = self.config.get('bm25_parameters', {}).get('top_k_candidates', 50)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include candidates with positive scores
                candidate_info = {
                    'index': idx,
                    'score': scores[idx],
                    'cui': self.combined_data.iloc[idx]['CUI'],
                    'system': self.combined_data.iloc[idx]['System'],
                    'tty': self.combined_data.iloc[idx]['TTY'],
                    'code': self.combined_data.iloc[idx]['CODE'],
                    'str': self.combined_data.iloc[idx]['STR'],
                    'sty': self.combined_data.iloc[idx]['STY'],
                    'cleaned_description': self.combined_data.iloc[idx]['cleaned_description'],
                    'tokens': self.combined_data.iloc[idx]['tokens'],
                    'source': self.combined_data.iloc[idx]['source'],
                    'source_index': self.combined_data.iloc[idx]['source_index']
                }
                candidates.append(candidate_info)
        
        return candidates
    
    def generate_initial_candidates(self, top_k: int = 50) -> Dict:
        """Generate initial BM25 candidates for all test descriptions"""
        
        if not self.bm25_model:
            raise ValueError("BM25 model not initialized. Call initialize_bm25() first.")
        
        logger.info(f"Generating initial candidates for {len(self.test_data)} test descriptions...")
        
        candidates_dict = {}
        
        for idx, row in self.test_data.iterrows():
            description = row['Input Entity Description']
            entity_type = row['Entity Type']
            tokens = row['tokens']
            
            # Get BM25 candidates with entity type
            candidates = self.get_bm25_candidates(tokens, entity_type=entity_type, top_k=top_k)
            
            candidates_dict[idx] = {
                'original_description': description,
                'entity_type': entity_type,
                'tokens': tokens,
                'candidates': candidates,
                'num_candidates': len(candidates)
            }
            
            if idx % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(self.test_data)} descriptions")
        
        logger.info("Initial candidate generation completed!")
        return candidates_dict
    
    def build_enhanced_graph_for_concept(self, candidate: Dict, max_related: int = None, expand_radius: int = None) -> nx.Graph:
        """Build an enhanced knowledge graph leveraging CUI, STY, TTY relationships and cross-system mappings"""
        
        G = nx.Graph()
        cui = candidate['cui']
        sty = candidate.get('sty', '')
        processed_cuis = set()
        
        # Get parameters from config if not specified
        if max_related is None:
            max_related = self.config.get('graph_building', {}).get('max_related_concepts', 100)
        if expand_radius is None:
            expand_radius = self.config.get('graph_building', {}).get('expand_radius', 2)
        
        # Multi-level expansion based on semantic relationships
        cuis_to_explore = [(cui, 0)]  # (cui, depth)
        
        while cuis_to_explore:
            current_cui, depth = cuis_to_explore.pop(0)
            
            if current_cui in processed_cuis or depth > expand_radius:
                continue
                
            processed_cuis.add(current_cui)
            
            # Get all concepts for this CUI from both systems
            rxnorm_concepts = self.rxnorm_data[self.rxnorm_data['CUI'] == current_cui]
            snomed_concepts = self.snomed_data[self.snomed_data['CUI'] == current_cui]
            all_concepts = pd.concat([rxnorm_concepts, snomed_concepts], ignore_index=True)
            
            # Add nodes for this CUI
            for idx, row in all_concepts.iterrows():
                node_id = f"{row['System']}_{row['CODE']}"
                if node_id not in G:
                    G.add_node(node_id,
                              cui=row['CUI'],
                              code=row['CODE'],
                              label=row['STR'],
                              system=row['System'],
                              tty=row['TTY'],
                              sty=row['STY'],
                              cleaned_description=row.get('cleaned_description', ''),
                              tokens=row.get('tokens', []),
                              depth=depth)
            
            # Find related CUIs for next level expansion (if within radius)
            if depth < expand_radius:
                # Pass entity type from candidate for better semantic alignment
                entity_type = candidate.get('entity_type', 'Unknown')
                related_cuis = self._find_semantically_related_cuis(current_cui, all_concepts, entity_type, max_related=10)
                for related_cui in related_cuis:
                    if related_cui not in processed_cuis:
                        cuis_to_explore.append((related_cui, depth + 1))
        
        # Add sophisticated edges based on multiple relationship types
        self._add_enhanced_edges(G)
        
        return G
    
    def _find_semantically_related_cuis(self, current_cui: str, current_concepts: pd.DataFrame, entity_type: str, max_related: int = 10) -> List[str]:
        """Find semantically related CUIs using STY, TTY, entity type alignment, and textual similarity"""
        
        if len(current_concepts) == 0:
            return []
        
        related_cuis = set()
        primary_sty = current_concepts['STY'].iloc[0]
        primary_tty = current_concepts['TTY'].iloc[0]
        
        # 1. Entity-type aligned STY filtering - prioritize semantically appropriate concepts
        entity_aligned_stys = self._get_entity_aligned_stys(entity_type)
        
        # If primary STY aligns with entity type, prioritize same STY
        if self._is_sty_aligned_with_entity(primary_sty, entity_type):
            same_sty_rxnorm = self.rxnorm_data[self.rxnorm_data['STY'] == primary_sty]['CUI'].unique()[:max_related//2]
            same_sty_snomed = self.snomed_data[self.snomed_data['STY'] == primary_sty]['CUI'].unique()[:max_related//2]
            related_cuis.update(same_sty_rxnorm)
            related_cuis.update(same_sty_snomed)
        else:
            # If primary STY doesn't align, look for better-aligned STYs
            for aligned_sty in entity_aligned_stys[:3]:  # Top 3 aligned STYs
                aligned_rxnorm = self.rxnorm_data[self.rxnorm_data['STY'] == aligned_sty]['CUI'].unique()[:max_related//6]
                aligned_snomed = self.snomed_data[self.snomed_data['STY'] == aligned_sty]['CUI'].unique()[:max_related//6]
                related_cuis.update(aligned_rxnorm)
                related_cuis.update(aligned_snomed)
        
        # 2. TTY-based relationships (e.g., ingredient to drug relationships)
        tty_related_cuis = self._find_tty_related_concepts(primary_tty, primary_sty, max_related//3)
        related_cuis.update(tty_related_cuis)
        
        # 3. Cross-system mappings with entity-type filtering
        if len(related_cuis) < max_related:
            cross_system_cuis = self._find_cross_system_concepts(primary_sty, max_related//3)
            # Filter cross-system concepts by entity-type alignment
            filtered_cross_system = self._filter_cuis_by_entity_alignment(cross_system_cuis, entity_type)
            related_cuis.update(filtered_cross_system)
        
        # Remove the current CUI and limit results
        related_cuis.discard(current_cui)
        return list(related_cuis)[:max_related]
    
    def _discover_entity_sty_relationships(self) -> Dict[str, List[str]]:
        """Discover which STYs are most associated with each entity type from actual test data"""
        
        if hasattr(self, '_entity_sty_cache'):
            return self._entity_sty_cache
        
        entity_sty_relationships = {}
        
        # We need to analyze what STYs appear most frequently for successful matches
        # For now, we'll use a frequency-based approach on the combined data
        
        # Get STY distributions by system (as proxy for entity type preferences)
        rxnorm_sty_counts = self.rxnorm_data['STY'].value_counts()
        snomed_sty_counts = self.snomed_data['STY'].value_counts()
        
        # Get top N STYs from config
        sty_config = self.config.get('sty_alignment', {})
        top_n_stys = sty_config.get('top_n_stys', 15)
        
        # Medicine entity type - prefer STYs common in RxNorm
        medicine_stys = rxnorm_sty_counts.head(top_n_stys).index.tolist()
        
        # Procedure/Lab/Diagnosis - prefer STYs common in SNOMED
        clinical_stys = snomed_sty_counts.head(top_n_stys).index.tolist()
        
        entity_sty_relationships = {
            'Medicine': medicine_stys,
            'Procedure': clinical_stys,
            'Diagnosis': clinical_stys,
            'Lab': clinical_stys
        }
        
        self._entity_sty_cache = entity_sty_relationships
        logger.info(f"Discovered STY relationships for {len(entity_sty_relationships)} entity types")
        
        return entity_sty_relationships
    
    def _get_entity_aligned_stys(self, entity_type: str) -> List[str]:
        """Get STYs that are aligned with the given entity type based on discovered patterns"""
        
        entity_sty_relationships = self._discover_entity_sty_relationships()
        return entity_sty_relationships.get(entity_type, [])
    
    def _is_sty_aligned_with_entity(self, sty: str, entity_type: str) -> bool:
        """Check if an STY is aligned with an entity type based on discovered relationships"""
        
        entity_aligned_stys = self._get_entity_aligned_stys(entity_type)
        return sty in entity_aligned_stys
    
    def _filter_cuis_by_entity_alignment(self, cuis: List[str], entity_type: str) -> List[str]:
        """Filter CUIs to keep only those with STYs aligned to the entity type"""
        
        filtered_cuis = []
        
        for cui in cuis:
            # Check STYs for this CUI in both systems
            rxnorm_stys = self.rxnorm_data[self.rxnorm_data['CUI'] == cui]['STY'].unique()
            snomed_stys = self.snomed_data[self.snomed_data['CUI'] == cui]['STY'].unique()
            all_stys = list(rxnorm_stys) + list(snomed_stys)
            
            # If any STY is aligned with entity type, keep this CUI
            if any(self._is_sty_aligned_with_entity(sty, entity_type) for sty in all_stys):
                filtered_cuis.append(cui)
        
        return filtered_cuis
    
    def _discover_tty_relationships(self) -> Dict[str, List[str]]:
        """Discover TTY relationships automatically from the data by analyzing CUI co-occurrences"""
        
        if hasattr(self, '_tty_relationships_cache'):
            return self._tty_relationships_cache
        
        tty_relationships = {}
        
        # Analyze CUIs that have multiple TTYs to discover relationships
        combined_data = pd.concat([self.rxnorm_data, self.snomed_data])
        cui_tty_groups = combined_data.groupby('CUI')['TTY'].apply(list).reset_index()
        
        # Count TTY co-occurrences within the same CUI
        tty_cooccurrence = {}
        
        for _, row in cui_tty_groups.iterrows():
            ttys = list(set(row['TTY']))  # Remove duplicates
            if len(ttys) > 1:
                for i, tty1 in enumerate(ttys):
                    for tty2 in ttys[i+1:]:
                        # Count bidirectional relationships
                        key1 = (tty1, tty2)
                        key2 = (tty2, tty1)
                        tty_cooccurrence[key1] = tty_cooccurrence.get(key1, 0) + 1
                        tty_cooccurrence[key2] = tty_cooccurrence.get(key2, 0) + 1
        
        # Build relationships based on co-occurrence frequency
        min_cooccurrence = self.config.get('graph_building', {}).get('min_tty_cooccurrence', 10)
        
        for (tty1, tty2), count in tty_cooccurrence.items():
            if count >= min_cooccurrence:
                if tty1 not in tty_relationships:
                    tty_relationships[tty1] = []
                if tty2 not in tty_relationships[tty1]:
                    tty_relationships[tty1].append(tty2)
        
        # Cache the results
        self._tty_relationships_cache = tty_relationships
        
        logger.info(f"Discovered TTY relationships for {len(tty_relationships)} term types")
        return tty_relationships
    
    def _find_tty_related_concepts(self, primary_tty: str, primary_sty: str, max_results: int) -> List[str]:
        """Find concepts with related term types based on discovered relationships"""
        
        # Use discovered relationships instead of hardcoded ones
        tty_relationships = self._discover_tty_relationships()
        related_ttys = tty_relationships.get(primary_tty, [])
        
        if not related_ttys:
            return []
        
        related_cuis = set()
        
        for related_tty in related_ttys:
            # Search in both systems for related TTY with same STY
            rxnorm_matches = self.rxnorm_data[
                (self.rxnorm_data['TTY'] == related_tty) & 
                (self.rxnorm_data['STY'] == primary_sty)
            ]['CUI'].unique()[:max_results//len(related_ttys)]
            
            snomed_matches = self.snomed_data[
                (self.snomed_data['TTY'] == related_tty) & 
                (self.snomed_data['STY'] == primary_sty)
            ]['CUI'].unique()[:max_results//len(related_ttys)]
            
            related_cuis.update(rxnorm_matches)
            related_cuis.update(snomed_matches)
        
        return list(related_cuis)[:max_results]
    
    def _find_cross_system_concepts(self, primary_sty: str, max_results: int) -> List[str]:
        """Find concepts that exist in both RxNorm and SNOMED (high-quality mappings)"""
        
        # Get CUIs that appear in both systems with the same STY
        rxnorm_cuis = set(self.rxnorm_data[self.rxnorm_data['STY'] == primary_sty]['CUI'].unique())
        snomed_cuis = set(self.snomed_data[self.snomed_data['STY'] == primary_sty]['CUI'].unique())
        
        # Cross-system CUIs are high-quality mappings
        cross_system_cuis = list(rxnorm_cuis & snomed_cuis)
        
        return cross_system_cuis[:max_results]
    
    def _add_enhanced_edges(self, G: nx.Graph):
        """Add sophisticated edges based on multiple relationship types"""
        
        nodes_list = list(G.nodes())
        
        for i, node1 in enumerate(nodes_list):
            for j, node2 in enumerate(nodes_list[i+1:], i+1):
                node1_data = G.nodes[node1]
                node2_data = G.nodes[node2]
                
                # Get edge weights from config
                edge_weights = self.config.get('edge_weights', {})
                
                # 1. Same CUI - strongest relationship (synonyms/variants)
                if node1_data['cui'] == node2_data['cui']:
                    edge_weight = edge_weights.get('same_cui', 1.0)
                    relation_type = 'same_cui'
                    
                    # Bonus for cross-system mappings (RxNorm ↔ SNOMED)
                    if node1_data['system'] != node2_data['system']:
                        edge_weight += edge_weights.get('cross_system_cui_bonus', 0.2)
                        relation_type = 'cross_system_cui'
                    
                    G.add_edge(node1, node2, relation_type=relation_type, weight=edge_weight)
                
                # 2. Same semantic type (STY) - moderate relationship
                elif node1_data['sty'] == node2_data['sty']:
                    weight = edge_weights.get('same_sty', 0.6)
                    G.add_edge(node1, node2, relation_type='same_semantic_type', weight=weight)
                
                # 3. TTY-based relationships (clinical relationships)
                elif self._are_tty_related(node1_data['tty'], node2_data['tty']):
                    weight = edge_weights.get('tty_related', 0.4)
                    G.add_edge(node1, node2, relation_type='tty_related', weight=weight)
                
                # 4. System-specific relationships
                elif node1_data['system'] == node2_data['system']:
                    # Same system but different CUI/STY - weak relationship
                    weight = edge_weights.get('same_system', 0.2)
                    G.add_edge(node1, node2, relation_type='same_system', weight=weight)
    
    def _are_tty_related(self, tty1: str, tty2: str) -> bool:
        """Check if two term types have clinical relationships based on discovered patterns"""
        
        # Use discovered relationships instead of hardcoded ones
        tty_relationships = self._discover_tty_relationships()
        
        # Check if tty2 is in the related list of tty1, or vice versa
        return (tty2 in tty_relationships.get(tty1, []) or 
                tty1 in tty_relationships.get(tty2, []))
    
    # Keep the original method for backward compatibility
    def build_graph_for_concept(self, candidate: Dict, max_related: int = 50) -> nx.Graph:
        """Build enhanced knowledge graph (wrapper for enhanced version)"""
        return self.build_enhanced_graph_for_concept(candidate, max_related, expand_radius=1)
    
    def get_enhanced_tty_weight(self, tty: str, entity_type: str, sty: str = None) -> float:
        """Enhanced TTY weighting based on entity type and semantic type"""
        
        # Get TTY weights from config
        tty_weights = self.config.get('tty_weights', {})
        entity_weights = tty_weights.get(entity_type, tty_weights.get('Diagnosis', {}))
        base_weight = entity_weights.get(tty, entity_weights.get('default', 1.0))
        
        # Semantic type bonus (general patterns, not hardcoded examples)
        sty_bonus = 1.0
        if sty:
            # Give bonus for STY that semantically aligns with entity type
            # This is a general approach, not tied to specific example STY values
            sty_lower = sty.lower()
            entity_lower = entity_type.lower()
            
            # General semantic alignment bonus
            if ('substance' in sty_lower or 'drug' in sty_lower or 'chemical' in sty_lower) and entity_lower == 'medicine':
                sty_bonus = 1.15
            elif ('disease' in sty_lower or 'disorder' in sty_lower or 'syndrome' in sty_lower) and entity_lower == 'diagnosis':
                sty_bonus = 1.15
            elif ('procedure' in sty_lower or 'therapeutic' in sty_lower) and entity_lower == 'procedure':
                sty_bonus = 1.15
            elif ('laboratory' in sty_lower or 'test' in sty_lower) and entity_lower == 'lab':
                sty_bonus = 1.15
        
        return base_weight * sty_bonus
    
    def get_tty_weight(self, tty: str, entity_type: str) -> float:
        """Backward compatibility wrapper"""
        return self.get_enhanced_tty_weight(tty, entity_type)
    
    def get_system_weight(self, system: str, entity_type: str) -> float:
        """Get weight for a system based on entity type"""
        
        # Get system weights from config
        system_weights = self.config.get('system_weights', {})
        entity_weights = system_weights.get(entity_type, {})
        return entity_weights.get(system, entity_weights.get('default', 1.0))
    
    def _get_sty_entity_alignment_score(self, sty: str, entity_type: str) -> float:
        """Calculate alignment score between STY and entity type based on discovered relationships"""
        
        if not sty:
            return 1.0  # Neutral score for missing STY
        
        # Get discovered STY rankings for this entity type
        entity_aligned_stys = self._get_entity_aligned_stys(entity_type)
        
        if sty in entity_aligned_stys:
            # Score based on ranking in the discovered list (higher rank = higher score)
            try:
                rank = entity_aligned_stys.index(sty)
                
                # Get scoring parameters from config
                sty_config = self.config.get('sty_alignment', {})
                max_score = sty_config.get('max_score', 1.0)
                min_score = sty_config.get('min_score', 0.1)
                score_decay = sty_config.get('score_decay', 0.9)
                
                # Calculate score with exponential decay
                score = max_score * (score_decay ** rank)
                return max(score, min_score)
                
            except ValueError:
                return sty_config.get('min_score', 0.1)
        
        # Not in aligned STYs - check if it appears in any system
        all_rxnorm_stys = set(self.rxnorm_data['STY'].unique())
        all_snomed_stys = set(self.snomed_data['STY'].unique())
        
        if sty in all_rxnorm_stys or sty in all_snomed_stys:
            return 2.0  # Valid STY but not aligned with entity type
        
        return 1.0  # Unknown STY
    
    def traverse_and_score_graph(self, G: nx.Graph, entity_type: str, original_tokens: List[str]) -> Dict:
        """Traverse the graph and score nodes based on relevance"""
        
        if len(G.nodes()) == 0:
            return {'best_node': None, 'score': 0.0, 'explanation': 'Empty graph'}
        
        node_scores = {}
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Base score components
            tty_weight = self.get_enhanced_tty_weight(node_data['tty'], entity_type, node_data.get('sty'))
            system_weight = self.get_system_weight(node_data['system'], entity_type)
            
            # Token overlap score (how many original tokens appear in this concept)
            node_tokens = node_data.get('tokens', [])
            if node_tokens:
                token_overlap = len(set(original_tokens) & set(node_tokens)) / len(set(original_tokens))
            else:
                token_overlap = 0.0
            
            # Centrality score (how well connected this node is)
            try:
                centrality = nx.degree_centrality(G)[node] if len(G) > 1 else 1.0
            except:
                centrality = 1.0
            
            # NEW: STY-Entity alignment score
            sty_alignment_score = self._get_sty_entity_alignment_score(node_data.get('sty', ''), entity_type)
            
            # Get scoring weights from config
            weights = self.config.get('graph_scoring_weights', {})
            
            # Enhanced scoring with configurable weights
            total_score = (tty_weight * weights.get('tty_weight', 0.35) +
                          system_weight * weights.get('system_weight', 0.15) +
                          token_overlap * weights.get('token_overlap', 0.25) +
                          centrality * weights.get('centrality', 0.10) +
                          sty_alignment_score * weights.get('sty_alignment', 0.15))
            
            node_scores[node] = {
                'total_score': total_score,
                'tty_weight': tty_weight,
                'system_weight': system_weight,
                'token_overlap': token_overlap,
                'centrality': centrality,
                'sty_alignment_score': sty_alignment_score,
                'node_data': node_data
            }
        
        # Find best node
        best_node = max(node_scores.keys(), key=lambda x: node_scores[x]['total_score'])
        best_score_info = node_scores[best_node]
        
        return {
            'best_node': best_node,
            'best_node_data': best_score_info['node_data'],
            'score': best_score_info['total_score'],
            'score_breakdown': {
                'tty_weight': best_score_info['tty_weight'],
                'system_weight': best_score_info['system_weight'],
                'token_overlap': best_score_info['token_overlap'],
                'centrality': best_score_info['centrality'],
                'sty_alignment_score': best_score_info['sty_alignment_score']
            },
            'graph_stats': {
                'num_nodes': len(G.nodes()),
                'num_edges': len(G.edges()),
                'connected_components': nx.number_connected_components(G)
            },
            'all_scores': node_scores
        }
    
    def enhance_candidates_with_graphs(self, candidates_dict: Dict, max_candidates_per_query: int = 10) -> Dict:
        """Enhance BM25 candidates with knowledge graph analysis"""
        
        logger.info("Enhancing candidates with knowledge graph analysis...")
        
        enhanced_results = {}
        
        for query_idx, query_data in candidates_dict.items():
            original_description = query_data['original_description']
            entity_type = query_data['entity_type']
            tokens = query_data['tokens']
            candidates = query_data['candidates'][:max_candidates_per_query]
            
            enhanced_candidates = []
            
            for i, candidate in enumerate(candidates):
                try:
                    # Add entity type to candidate for graph building
                    candidate_with_entity = {**candidate, 'entity_type': entity_type}
                    
                    # Build local graph for this candidate
                    graph = self.build_graph_for_concept(candidate_with_entity)
                    
                    # Score the graph
                    graph_result = self.traverse_and_score_graph(graph, entity_type, tokens)
                    
                    # Combine BM25 score with graph score using config weights
                    bm25_score = candidate['score']
                    graph_score = graph_result['score']
                    
                    final_weights = self.config.get('final_score_weights', {})
                    bm25_weight = final_weights.get('bm25_weight', 0.6)
                    graph_weight = final_weights.get('graph_weight', 0.4)
                    
                    combined_score = (bm25_score * bm25_weight + graph_score * graph_weight)
                    
                    enhanced_candidate = {
                        **candidate,
                        'graph_analysis': graph_result,
                        'combined_score': combined_score,
                        'graph_enhanced': True
                    }
                    
                    enhanced_candidates.append(enhanced_candidate)
                    
                except Exception as e:
                    logger.warning(f"Failed to build graph for candidate {i}: {str(e)}")
                    # Keep original candidate without graph enhancement
                    enhanced_candidate = {
                        **candidate,
                        'graph_analysis': None,
                        'combined_score': candidate['score'],
                        'graph_enhanced': False
                    }
                    enhanced_candidates.append(enhanced_candidate)
            
            # Re-sort candidates by combined score
            enhanced_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            enhanced_results[query_idx] = {
                **query_data,
                'enhanced_candidates': enhanced_candidates,
                'num_enhanced': len([c for c in enhanced_candidates if c['graph_enhanced']])
            }
            
            if query_idx % 10 == 0:
                logger.info(f"Enhanced {query_idx + 1} queries with knowledge graphs")
        
        logger.info("Knowledge graph enhancement completed!")
        return enhanced_results
    
    def get_best_match_for_description(self, description: str, entity_type: str, top_k: int = 10) -> Dict:
        """Get the best matching standardized code for a clinical description"""
        
        # Preprocess the description
        processed = self.preprocessor.preprocess_descriptions([description])
        if not processed[0] or not processed[0][0]['tokens']:
            return {
                'success': False,
                'error': 'Could not process description',
                'description': description
            }
        
        tokens = processed[0][0]['tokens']
        
        try:
            # Step 1: Get BM25 candidates
            candidates = self.get_bm25_candidates(tokens, top_k=top_k)
            
            if not candidates:
                return {
                    'success': False,
                    'error': 'No BM25 candidates found',
                    'description': description
                }
            
            # Step 2: Enhance candidates with knowledge graphs
            candidates_dict = {0: {
                'original_description': description,
                'entity_type': entity_type,
                'tokens': tokens,
                'candidates': candidates,
                'num_candidates': len(candidates)
            }}
            
            enhanced_results = self.enhance_candidates_with_graphs(candidates_dict, max_candidates_per_query=5)
            enhanced_candidates = enhanced_results[0]['enhanced_candidates']
            
            # Step 3: Get the best candidate
            if enhanced_candidates:
                best_candidate = enhanced_candidates[0]
                
                # Determine the output system and format the result
                output_system = 'RXNORM' if best_candidate['system'] == 'RXNORM' else 'SNOMEDCT_US'
                
                return {
                    'success': True,
                    'original_description': description,
                    'entity_type': entity_type,
                    'output_coding_system': output_system,
                    'output_target_code': str(best_candidate['code']),
                    'output_target_description': best_candidate['str'],
                    'bm25_score': best_candidate['score'],
                    'combined_score': best_candidate['combined_score'],
                    'graph_enhanced': best_candidate['graph_enhanced'],
                    'cui': best_candidate['cui'],
                    'tty': best_candidate['tty'],
                    'sty': best_candidate['sty']
                }
            else:
                return {
                    'success': False,
                    'error': 'No enhanced candidates available',
                    'description': description
                }
                
        except Exception as e:
            logger.error(f"Error processing description '{description}': {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'description': description
            }
    
    def generate_predictions_for_all(self, max_candidates: int = 10) -> pd.DataFrame:
        """Generate predictions for all test descriptions"""
        
        logger.info(f"Generating predictions for all {len(self.test_data)} test descriptions...")
        
        results = []
        
        for idx, row in self.test_data.iterrows():
            description = row['Input Entity Description']
            entity_type = row['Entity Type']
            
            logger.info(f"Processing {idx + 1}/{len(self.test_data)}: '{description}'")
            
            # Get best match
            result = self.get_best_match_for_description(description, entity_type, top_k=max_candidates)
            
            if result['success']:
                results.append({
                    'Input Entity Description': description,
                    'Entity Type': entity_type,
                    'Output Coding System': result['output_coding_system'],
                    'Output Target Code': result['output_target_code'],
                    'Output Target Description': result['output_target_description'],
                    'BM25 Score': result['bm25_score'],
                    'Combined Score': result['combined_score'],
                    'Graph Enhanced': result['graph_enhanced'],
                    'CUI': result['cui'],
                    'TTY': result['tty'],
                    'STY': result['sty']
                })
            else:
                logger.warning(f"Failed to process '{description}': {result.get('error', 'Unknown error')}")
                results.append({
                    'Input Entity Description': description,
                    'Entity Type': entity_type,
                    'Output Coding System': 'UNKNOWN',
                    'Output Target Code': 'UNKNOWN',
                    'Output Target Description': 'UNKNOWN',
                    'BM25 Score': 0.0,
                    'Combined Score': 0.0,
                    'Graph Enhanced': False,
                    'CUI': 'UNKNOWN',
                    'TTY': 'UNKNOWN',
                    'STY': 'UNKNOWN'
                })
        
        results_df = pd.DataFrame(results)
        logger.info("Prediction generation completed!")
        
        return results_df
    
    def save_results_to_excel(self, results_df: pd.DataFrame, output_file: str = "Test_with_predictions.xlsx"):
        """Save results to Excel file in the required format"""
        
        logger.info(f"Saving results to {output_file}...")
        
        # Create the output dataframe with required columns
        output_df = pd.DataFrame({
            'Input Entity Description': results_df['Input Entity Description'],
            'Entity Type': results_df['Entity Type'],
            'Output Coding System': results_df['Output Coding System'],
            'Output Target Code': results_df['Output Target Code'],
            'Output Target Description': results_df['Output Target Description']
        })
        
        # Save to Excel
        output_df.to_excel(output_file, index=False)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary statistics
        print(f"\n=== Results Summary ===")
        print(f"Total predictions: {len(output_df)}")
        print(f"Coding system distribution:")
        print(output_df['Output Coding System'].value_counts())
        print(f"Entity type distribution:")
        print(output_df['Entity Type'].value_counts())
        
        # Show some sample predictions
        print(f"\n=== Sample Predictions ===")
        for i in range(min(5, len(output_df))):
            row = output_df.iloc[i]
            print(f"{i+1}. '{row['Input Entity Description']}' ({row['Entity Type']})")
            print(f"   -> {row['Output Coding System']}: {row['Output Target Code']} - {row['Output Target Description']}")
        
        return output_file


def main(max_test_cases: int = None, config_path: str = 'scoring_config.yaml'):
    """Main function to demonstrate the complete Knowledge Graph Navigator pipeline
    
    Args:
        max_test_cases: Maximum number of test cases to process (None = all 400)
        config_path: Path to the YAML configuration file
    """
    
    # Initialize the Knowledge Graph Navigator with config
    navigator = KnowledgeGraphNavigator(config_path)
    
    # Step 1: Load the data
    navigator.load_data()
    
    # Step 2: Preprocess clinical data
    logger.info("Preprocessing clinical data...")
    
    # Process test data
    test_descriptions = navigator.test_data['Input Entity Description'].tolist()
    navigator.processed_test, test_expansions = navigator.preprocessor.preprocess_descriptions(test_descriptions)
    navigator.test_data['cleaned_description'] = [item['cleaned'] for item in navigator.processed_test]
    navigator.test_data['tokens'] = [item['tokens'] for item in navigator.processed_test]
    navigator.test_data['token_string'] = [item['token_string'] for item in navigator.processed_test]
    
    print(f"\n=== Preprocessed {len(test_expansions)} abbreviation expansions in test data ===")
    
    # Process target data (RxNorm and SNOMED)
    logger.info("Preprocessing target descriptions (RxNorm and SNOMED)...")
    
    # Process RxNorm descriptions
    rxnorm_descriptions = navigator.rxnorm_data['STR'].tolist()
    navigator.processed_rxnorm, _ = navigator.preprocessor.preprocess_descriptions(rxnorm_descriptions)
    navigator.rxnorm_data['cleaned_description'] = [item['cleaned'] for item in navigator.processed_rxnorm]
    navigator.rxnorm_data['tokens'] = [item['tokens'] for item in navigator.processed_rxnorm]
    navigator.rxnorm_data['token_string'] = [item['token_string'] for item in navigator.processed_rxnorm]
    
    # Process SNOMED descriptions  
    snomed_descriptions = navigator.snomed_data['STR'].tolist()
    navigator.processed_snomed, _ = navigator.preprocessor.preprocess_descriptions(snomed_descriptions)
    navigator.snomed_data['cleaned_description'] = [item['cleaned'] for item in navigator.processed_snomed]
    navigator.snomed_data['tokens'] = [item['tokens'] for item in navigator.processed_snomed]
    navigator.snomed_data['token_string'] = [item['token_string'] for item in navigator.processed_snomed]
    
    # Step 3: Initialize BM25 model
    navigator.initialize_bm25()
    
    # Step 4: Test with a few examples first
    print("\n=== Testing Individual Predictions ===")
    
    # Test with first 5 test cases
    for i in range(min(5, len(navigator.test_data))):
        row = navigator.test_data.iloc[i]
        description = row['Input Entity Description']
        entity_type = row['Entity Type']
        
        print(f"\nTest Case {i+1}: '{description}' ({entity_type})")
        
        # Get best match
        result = navigator.get_best_match_for_description(description, entity_type)
        
        if result['success']:
            print(f"✓ SUCCESS: {result['output_coding_system']} - {result['output_target_code']}")
            print(f"  Description: {result['output_target_description']}")
            print(f"  Scores: BM25={result['bm25_score']:.2f}, Combined={result['combined_score']:.2f}")
            print(f"  Graph Enhanced: {result['graph_enhanced']}")
        else:
            print(f"✗ FAILED: {result.get('error', 'Unknown error')}")
        
        print("---")
    
    # Step 5: Generate predictions for test cases
    if max_test_cases:
        print(f"\n=== Generating Predictions for First {max_test_cases} Test Cases ===")
        navigator.test_data = navigator.test_data.head(max_test_cases)
        print(f"Processing {len(navigator.test_data)} test cases for demonstration...")
    else:
        print(f"\n=== Generating Predictions for ALL {len(navigator.test_data)} Test Cases ===")
        print("This may take 30-60 minutes depending on your system...")
        print("Processing all 400 test cases...")
    
    results_df = navigator.generate_predictions_for_all(max_candidates=10)
    
    # Step 6: Save results to Excel
    output_file = navigator.save_results_to_excel(results_df, "Test_with_predictions.xlsx")
    
    print(f"\n=== Knowledge Graph Navigator Pipeline Complete! ===")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results_df)} clinical descriptions")
    
    # Show detailed results for first few predictions
    print(f"\n=== Detailed Results Sample ===")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        print(f"{i+1}. Input: '{row['Input Entity Description']}' ({row['Entity Type']})")
        print(f"   Output: {row['Output Coding System']} - {row['Output Target Code']}")
        print(f"   Description: {row['Output Target Description']}")
        print(f"   CUI: {row['CUI']}, TTY: {row['TTY']}")
        print(f"   Scores: BM25={row['BM25 Score']:.2f}, Combined={row['Combined Score']:.2f}")
        print(f"   Graph Enhanced: {row['Graph Enhanced']}")
        print()


def demo_mode():
    """Run a quick demo with just a few test cases"""
    
    print("=== Knowledge Graph Navigator - Demo Mode ===")
    print("Processing first 5 test cases only...\n")
    
    navigator = KnowledgeGraphNavigator()
    navigator.load_data()
    
    # Quick preprocessing for demo
    test_descriptions = navigator.test_data['Input Entity Description'].head(5).tolist()
    navigator.processed_test, _ = navigator.preprocessor.preprocess_descriptions(test_descriptions)
    navigator.test_data = navigator.test_data.head(5).copy()
    navigator.test_data['tokens'] = [item['tokens'] for item in navigator.processed_test]
    
    # Process subset of target data for demo
    navigator.rxnorm_data = navigator.rxnorm_data.head(50000)  # Use subset for speed
    navigator.snomed_data = navigator.snomed_data.head(50000)  # Use subset for speed
    
    rxnorm_descriptions = navigator.rxnorm_data['STR'].tolist()
    navigator.processed_rxnorm, _ = navigator.preprocessor.preprocess_descriptions(rxnorm_descriptions)
    navigator.rxnorm_data['tokens'] = [item['tokens'] for item in navigator.processed_rxnorm]
    
    snomed_descriptions = navigator.snomed_data['STR'].tolist()
    navigator.processed_snomed, _ = navigator.preprocessor.preprocess_descriptions(snomed_descriptions)
    navigator.snomed_data['tokens'] = [item['tokens'] for item in navigator.processed_snomed]
    
    # Initialize BM25 and generate predictions
    navigator.initialize_bm25()
    results_df = navigator.generate_predictions_for_all(max_candidates=5)
    navigator.save_results_to_excel(results_df, "Demo_predictions.xlsx")
    
    print("\n=== Demo Complete! ===")


if __name__ == "__main__":
    # Choose which mode to run:
    
    # Option 1: Process ALL 400 test cases (production mode)
    main()  # This will take 30-60 minutes
    
    # Option 2: Process first 20 test cases (demo mode)
    # main(max_test_cases=20)  # Much faster for demonstration
    
    # Option 3: Quick demo with subset of data (fastest)
    # demo_mode()
