https://github.com/tejasexpress/Harmonizing-Clinical-Concepts

Repository Link

# Clinical Data Harmonization: Multiple Approaches

## Overview

This repository contains **three different approaches** for harmonizing clinical data by mapping free-text clinical descriptions to standardized codes from RxNorm and SNOMED CT. The approaches are presented in order of performance:

1. **🏆 Knowledge Graph Navigator** (This Implementation) - **Best Performing**
2. **🔍 Classical Search Engine** (Alternative Approach) 
3. **🤖 Transformer-Based Approach** (Alternative Approach)

---

## 🏆 Approach #1: Knowledge Graph Navigator (Best Performing)

### What This Approach Does

The **Knowledge Graph Navigator** is our **highest-performing** clinical concept harmonization system. It uses a sophisticated hybrid approach that combines:

- **Fast text matching** (BM25) for initial candidate generation
- **Semantic knowledge graphs** for context-aware understanding  
- **Entity-type awareness** for targeted matching
- **Data-driven relationship discovery** instead of hardcoded rules

### Repository Structure

```
project_root/
├── knowledge_graph_navigator.py    # Main implementation (this approach)
├── scoring_config.yaml            # All scoring parameters (configurable)
├── Test.xlsx                     # Input clinical descriptions
├── Target Description Files/
│   ├── rxnorm_all_data.parquet   # RxNorm terminology (379K concepts)
│   ├── snomed_all_data.parquet   # SNOMED CT terminology (1M+ concepts)
│   └── Column Reference Guide.md # Data schema documentation
└── README.md                     # This documentation
```

### How It Works: Quick Walkthrough

1. **📝 Preprocessing**: Cleans text, expands medical abbreviations ("mri" → "magnetic resonance imaging"), preserves dosages ("500 mg")

2. **🔍 Initial Matching**: Uses BM25 to quickly find ~50 candidate concepts from 1.4M+ total concepts, incorporating entity type into the search

3. **🕸️ Graph Construction**: For each candidate, builds a small knowledge graph by:
   - Finding related concepts via CUI (Concept Unique Identifier) relationships
   - Connecting concepts through STY (Semantic Type) and TTY (Term Type) relationships
   - Learning these relationships from the data automatically

4. **⚖️ Multi-Component Scoring**: Scores each concept using:
   - **TTY Weight (35%)**: How appropriate is this term type for the entity?
   - **Token Overlap (25%)**: Direct text similarity
   - **STY Alignment (15%)**: Does the semantic type match the entity type?
   - **System Weight (15%)**: Is this the right terminology system (RxNorm vs SNOMED)?
   - **Centrality (10%)**: How well-connected is this concept?

5. **🎯 Final Selection**: Combines BM25 score (60%) + Graph score (40%) to select the best match

### Why This Approach Works Best

- ✅ **Entity-Type Aware**: Knows that "Medicine" should prefer RxNorm and SCD term types
- ✅ **Preserves Clinical Context**: Handles "0.1 mg/ml" and medical abbreviations correctly  
- ✅ **Data-Driven**: Learns relationships from the actual dataset, not hardcoded rules
- ✅ **Configurable**: All parameters in `scoring_config.yaml` for easy tuning
- ✅ **Cross-System**: Leverages both RxNorm and SNOMED strengths

---

## 🚀 Setup and Execution (Knowledge Graph Navigator)

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for loading large vocabularies)
- Required files:
  - `Test.xlsx`
  - `Target Description Files/rxnorm_all_data.parquet`
  - `Target Description Files/snomed_all_data.parquet`

### Quick Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install pandas networkx rank_bm25 openpyxl pyarrow pyyaml scikit-learn
```

### Execution Options

#### Option 1: Quick Test (Single Query)
```python
from knowledge_graph_navigator import KnowledgeGraphNavigator

# Initialize and setup
navigator = KnowledgeGraphNavigator('scoring_config.yaml')
navigator.load_data()
navigator.preprocess_clinical_data()
navigator.initialize_bm25()

# Test single description
result = navigator.get_best_match_for_description("Paracetamol 500 mg", "Medicine")
print(f"Best match: {result['output_target_description']}")
```

#### Option 2: Process All 400 Test Cases (Production)
```python
from knowledge_graph_navigator import main
main()  # Takes 30-60 minutes
```

#### Option 3: Demo Mode (First 20 Cases)
```python
from knowledge_graph_navigator import main
main(max_test_cases=20)  # Much faster for demonstration
```

#### Option 4: Command Line
```bash
python knowledge_graph_navigator.py
```

### Expected Output
- **Test_with_predictions.xlsx**: Final results in required format
- **Console progress**: Shows processing status and sample results
- **Performance metrics**: BM25 scores, combined scores, success rates

### Sample Results
```
Input: 'Paracetamol 500 mg' (Medicine)
Output: SNOMEDCT_US - 322236009
Description: Paracetamol 500 mg oral tablet
Score: 17.08 (BM25: 14.77, Graph: 2.31)
```

---

## 📊 Technical Details (Knowledge Graph Navigator)

### Algorithm Overview

**Scoring Formula:**
```
Final Score = (BM25_Score × 0.6) + (Graph_Score × 0.4)

Graph_Score = (TTY_Weight × 0.35) + 
              (Token_Overlap × 0.25) + 
              (STY_Alignment × 0.15) + 
              (System_Weight × 0.15) + 
              (Centrality × 0.10)
```

### Key Features
- **Entity-Type Specific Scoring**: Different weights for Medicine, Procedure, Diagnosis, Lab
- **Data-Driven Discovery**: Automatically learns TTY relationships and STY alignments
- **Configurable Parameters**: All scoring weights in `scoring_config.yaml`
- **Cross-System Integration**: Seamlessly combines RxNorm and SNOMED CT

### Performance Characteristics
- **Speed**: ~400 queries in 30-60 minutes
- **Memory**: Peak usage ~6-8 GB during processing
- **Accuracy**: High precision through knowledge graph enhancement

---

## 🔍 Approach #2: Classical Search Engine

### What This Approach Does

The **Classical Search Engine** approach represents a hybrid search and re-ranking system that combines traditional lexical search with modern semantic search techniques. This approach was developed as an alternative to pure keyword matching but was ultimately outperformed by the Knowledge Graph Navigator.

### How It Works

1. **Lexical Search**: Uses BM25Okapi algorithm for fast keyword-based candidate identification from the large terminology space

2. **Semantic Search**: Employs SapBERT sentence transformer model (pre-trained on biomedical text) to generate dense vector representations of clinical descriptions, enabling semantic similarity matching beyond keyword overlap

3. **FAISS Indexing**: Utilizes Facebook AI Similarity Search (FAISS) library for efficient similarity search and clustering over millions of concept vectors

4. **Cross-Encoder Re-ranking**: After retrieving candidates from both lexical and semantic stages, applies a cross-encoder model to re-rank candidates by computing fine-grained similarity scores between input descriptions and candidate descriptions

### Why Classical Search Has Limitations

Traditional search engines rely primarily on inverted indexes and keyword matching, which is insufficient for clinical data nuances:

- **Surface-Level Matching**: Cannot understand that "brachialgia" is a synonym for "arm pain"
- **Lack of Clinical Context**: Fails to differentiate between clinically distinct concepts like "fractured arm pain" vs "sore arm muscle"
- **No Semantic Understanding**: Operates on keyword matching without grasping hierarchical relationships in clinical terminologies
- **Precision Requirements**: Clinical concept harmonization requires understanding subtle but critical differences between terms, where misunderstandings can impact patient care

### Performance vs Knowledge Graph Navigator

While the Classical Search Engine approach incorporates semantic understanding through transformers and sophisticated re-ranking, it lacks the **entity-type awareness** and **data-driven relationship discovery** that makes the Knowledge Graph Navigator superior for clinical concept harmonization tasks.

---

## 🤖 Approach #3: Transformer-Based Approach (Future Implementation)

This approach would leverage state-of-the-art transformer models specifically fine-tuned for clinical concept harmonization. It represents a potential future direction for even more sophisticated semantic understanding of clinical text.

---

## 🎯 Conclusion

The **Knowledge Graph Navigator** represents the current state-of-the-art approach in this repository, combining the speed of traditional search with the semantic understanding of modern NLP, enhanced by domain-specific knowledge graph construction and entity-type awareness. This hybrid approach achieves superior performance by understanding both the textual and conceptual relationships inherent in clinical terminologies.

For production use, we recommend the Knowledge Graph Navigator approach due to its:
- Superior accuracy through multi-component scoring
- Configurable parameters for different use cases
- Data-driven intelligence that adapts to the actual terminology relationships
- Entity-type awareness that improves precision for different clinical domains