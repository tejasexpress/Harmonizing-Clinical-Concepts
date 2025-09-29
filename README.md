# Knowledge Graph Navigator for Clinical Data Harmonization

## Overview

The **Knowledge Graph Navigator** is a clinical concept harmonization system that maps input clinical descriptions to standardized codes from RxNorm and SNOMED CT. It combines text matching (BM25) with semantic understanding through knowledge graphs to achieve high-accuracy mappings.

## Solution Architecture

### Core Approach
The system uses a **hybrid approach** combining:
1. **Initial Text Matching**: BM25 (Okapi) for fast candidate generation
2. **Semantic Enhancement**: Knowledge graphs for context-aware scoring
3. **Entity-Type Awareness**: Using clinical entity types (Medicine, Procedure, Diagnosis, Lab) for targeted matching
4. **Data-Driven Intelligence**: Automatically discovers relationships from the dataset rather than using hardcoded rules

### Key Components

#### 1. Clinical Data Preprocessor
- **Text Cleaning**: Handles medical abbreviations, preserves numbers/units/percentages
- **Abbreviation Expansion**: Converts medical abbreviations (e.g., "mri" → "magnetic resonance imaging")
- **Number Preservation**: Maintains dosages and measurements (e.g., "0.1 mg/ml", "500 mg")
- **Tokenization**: Creates searchable tokens while removing stop words

#### 2. BM25 Candidate Generation
- **Fast Initial Matching**: Uses BM25Okapi algorithm for text similarity
- **Entity-Type Integration**: Incorporates entity type into search queries
- **Configurable Parameters**: Top-k candidates, scoring thresholds
- **Cross-System Search**: Searches both RxNorm and SNOMED simultaneously

#### 3. Knowledge Graph Construction
- **Multi-Level Expansion**: Builds graphs around candidate concepts using CUI relationships
- **Semantic Relationships**: Connects concepts via STY (Semantic Type) and TTY (Term Type)
- **Cross-System Mappings**: Links RxNorm and SNOMED concepts with shared CUIs
- **Data-Driven Discovery**: Automatically learns TTY and STY relationships from the dataset

#### 4. Graph-Based Scoring
- **Multi-Component Scoring**:
  - **TTY Weight (35%)**: Term type importance based on entity type
  - **Token Overlap (25%)**: Direct text similarity
  - **STY Alignment (15%)**: Semantic type relevance to entity type
  - **System Weight (15%)**: Preference for appropriate terminology system
  - **Centrality (10%)**: Graph connectivity importance
- **Configurable Weights**: All scoring parameters externalized to YAML configuration

#### 5. Final Score Combination
- **BM25 Score (60%)**: Text matching component
- **Graph Score (40%)**: Semantic understanding component
- **Configurable Balance**: Weights adjustable via configuration

## Dataset Structure

### Input Data
- **Test.xlsx**: Clinical descriptions to be harmonized
  - `Input Entity Description`: Free-text clinical descriptions
  - `Entity Type`: Medicine, Procedure, Diagnosis, or Lab

### Target Vocabularies
- **rxnorm_all_data.parquet**: RxNorm terminology (379,991 concepts)
- **snomed_all_data.parquet**: SNOMED CT terminology (1,035,233 concepts)

### Data Schema
Each concept contains:
- **CUI**: Concept Unique Identifier (links synonymous terms)
- **System**: Source vocabulary (RXNORM, SNOMEDCT_US)
- **TTY**: Term Type (role within vocabulary)
- **CODE**: Original vocabulary code
- **STR**: Human-readable term
- **STY**: Semantic Type (broad category)

## Key Features

### 1. Entity-Type Aware Matching
- **Medicine**: Prioritizes RxNorm, favors SCD/SBD term types
- **Procedure**: Prioritizes SNOMED, favors PT (Preferred Term)
- **Diagnosis**: Prioritizes SNOMED, favors PT term types
- **Lab**: Prioritizes SNOMED, includes LOINC terms

### 2. Intelligent Preprocessing
- **Medical Abbreviation Expansion**: 21+ common abbreviations
- **Number/Unit Preservation**: Maintains "0.1 mg/ml", "500 mg", "0.01%"
- **Parentheses Handling**: Extracts content while removing brackets

### 3. Data-Driven Relationship Discovery
- **TTY Relationships**: Learns co-occurrence patterns within CUIs
- **STY-Entity Alignment**: Discovers which semantic types best match entity types
- **Cross-System Mappings**: Identifies high-quality RxNorm↔SNOMED links

### 4. Configurable Scoring System
All heuristic parameters externalized to `scoring_config.yaml`:
- Final score combination weights
- Graph component weights
- TTY weights by entity type
- System preferences
- Edge weights in knowledge graphs

## Setup Instructions

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for loading large vocabularies)
- Required files:
  - `Test.xlsx`
  - `Target Description Files/rxnorm_all_data.parquet`
  - `Target Description Files/snomed_all_data.parquet`

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install pandas networkx rank_bm25 openpyxl pyarrow pyyaml scikit-learn
```

### Step 3: Verify Data Files
Ensure your directory structure looks like:
```
project_root/
├── knowledge_graph_navigator.py
├── scoring_config.yaml
├── Test.xlsx
└── Target Description Files/
    ├── rxnorm_all_data.parquet
    ├── snomed_all_data.parquet
    └── Column Reference Guide.md
```

### Step 4: Configuration
The system uses `scoring_config.yaml` for all parameters. Default configuration is provided, but you can modify:
- Score combination weights
- TTY importance by entity type
- System preferences
- Graph building parameters

## Usage

### Quick Start
```python
from knowledge_graph_navigator import KnowledgeGraphNavigator

# Initialize with default configuration
navigator = KnowledgeGraphNavigator('scoring_config.yaml')

# Load data
navigator.load_data()

# Preprocess clinical data
navigator.preprocess_clinical_data()

# Initialize BM25 model
navigator.initialize_bm25()

# Test single description
result = navigator.get_best_match_for_description(
    "Paracetamol 500 mg", 
    "Medicine"
)

print(f"Best match: {result['output_target_description']}")
print(f"System: {result['output_coding_system']}")
print(f"Code: {result['output_target_code']}")
```

### Running the Complete Pipeline

#### Option 1: Process All 400 Test Cases (Production)
```python
from knowledge_graph_navigator import main

# Process all test cases (30-60 minutes)
main()
```

#### Option 2: Demo Mode (First 20 Cases)
```python
from knowledge_graph_navigator import main

# Process first 20 cases for demonstration
main(max_test_cases=20)
```

#### Option 3: Command Line Execution
```bash
python knowledge_graph_navigator.py
```

### Expected Output
The system generates:
- **Test_with_predictions.xlsx**: Results in required format
- **Console output**: Progress updates, sample results, statistics
- **Performance metrics**: BM25 scores, combined scores, success rates

### Sample Results
```
Input: 'Paracetamol 500 mg' (Medicine)
Output: SNOMEDCT_US - 322236009
Description: Paracetamol 500 mg oral tablet
Score: 17.08 (BM25: 14.77, Graph: 2.31)
```

## Configuration

### scoring_config.yaml Structure
```yaml
# Final score combination
final_score_weights:
  bm25_weight: 0.6      # Text matching importance
  graph_weight: 0.4     # Semantic graph importance

# Graph scoring components
graph_scoring_weights:
  tty_weight: 0.35      # Term type importance
  system_weight: 0.15   # System preference
  token_overlap: 0.25   # Text similarity
  centrality: 0.10      # Graph connectivity
  sty_alignment: 0.15   # Semantic alignment

# TTY weights by entity type
tty_weights:
  Medicine:
    SCD: 5.0    # Semantic Clinical Drug (highest)
    SBD: 4.5    # Semantic Branded Drug
    IN: 3.5     # Ingredient
    # ... more TTY weights
```

### Customizing the Algorithm
1. **Modify Weights**: Edit `scoring_config.yaml`
2. **Add Abbreviations**: Update `medical_abbreviations` section
3. **Tune BM25**: Adjust `bm25_parameters`
4. **Graph Building**: Modify `graph_building` parameters

## Algorithm Details

### Scoring Formula
```
Final Score = (BM25_Score × 0.6) + (Graph_Score × 0.4)

Graph_Score = (TTY_Weight × 0.35) + 
              (Token_Overlap × 0.25) + 
              (STY_Alignment × 0.15) + 
              (System_Weight × 0.15) + 
              (Centrality × 0.10)
```

### Knowledge Graph Construction
1. **Start with candidate concept**
2. **Expand by CUI relationships** (radius = 2 levels)
3. **Add nodes**: All terms sharing CUIs
4. **Create edges**: Based on CUI, STY, TTY, and system relationships
5. **Score nodes**: Multi-component relevance scoring

### Data-Driven Intelligence
- **TTY Relationships**: Discovered by analyzing CUI co-occurrence (min 10 occurrences)
- **STY-Entity Mapping**: Top 15 STYs per entity type from frequency analysis
- **Cross-System Quality**: Concepts appearing in both RxNorm and SNOMED

## Performance & Troubleshooting

### Performance Characteristics
- **Accuracy**: High precision through knowledge graph enhancement
- **Speed**: ~400 queries in 30-60 minutes
- **Memory**: Peak usage ~6-8 GB during processing

### Common Issues & Solutions
1. **Memory Errors**: Increase system RAM or reduce dataset size
2. **Missing Dependencies**: `pip install pyarrow pyyaml`
3. **File Not Found**: Ensure all data files are in correct directories
4. **No Matches**: Check entity_type parameter is being passed correctly

### Debug Mode
```python
import logging
logging.getLogger().setLevel(logging.INFO)
```
