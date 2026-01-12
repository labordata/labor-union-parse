# Labor Union Parser

Extract affiliation and local designation from labor union name strings.

Given an input like `"SEIU Local 1199"`, the parser returns:
- **is_union**: `True` (detected as a union)
- **Affiliation**: `SEIU` (Service Employees International Union)
- **Designation**: `1199` (local number)

## Installation

```bash
pip install -e .
```

## Usage

### Python API

```python
from labor_union_parser import Extractor

extractor = Extractor()
result = extractor.extract("SEIU Local 1199")
print(result)
# {'is_union': True, 'union_score': 0.999, 'affiliation': 'SEIU',
#  'affiliation_unrecognized': False, 'designation': '1199', 'aff_score': 0.997}
```

For batch processing:

```python
from labor_union_parser import Extractor

extractor = Extractor()
results = extractor.extract_batch([
    "SEIU Local 1199",
    "Teamsters Local 705",
    "UAW Local 600",
])
```

For large datasets, use `extract_all` which yields results as a generator:

```python
from labor_union_parser import Extractor

extractor = Extractor()

# Process large list with progress bar
for result in extractor.extract_all(union_names, show_progress=True):
    print(result)

# Adjust batch size for memory/speed tradeoff
results = list(extractor.extract_all(union_names, batch_size=512))
```

### Filing Number Lookup

Look up OLMS filing numbers for a given affiliation and designation:

```python
from labor_union_parser import lookup_fnum

fnums = lookup_fnum("SEIU", "1199")
# [31847, 69557, 508557, ...]
```

### Command Line

```bash
# Process CSV file
labor-union-parser unions.csv -c union_name -o results.csv

# Process from stdin
echo "SEIU Local 1199" | labor-union-parser --no-header
# text,pred_is_union,pred_aff,pred_unknown,pred_desig,pred_union_score,pred_fnum,pred_fnum_multiple
# SEIU Local 1199,True,SEIU,False,1199,0.9992,"[31847, 69557, ...]",True
```

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `union_score` | Similarity score to union centroid (0-1) |
| `affiliation` | Predicted affiliation abbreviation (e.g., "SEIU", "IBT") or `None` |
| `affiliation_unrecognized` | `True` if detected as union but affiliation unrecognized |
| `designation` | Extracted local number (e.g., "1199") or empty string |
| `aff_score` | Similarity to nearest affiliation centroid (higher = more confident) |

## Training

Training data is in `training/data/labeled_data.csv` with columns:
- `text`: Union name string
- `aff_abbr`: Affiliation abbreviation (e.g., "SEIU", "IBT", "UAW")
- `desig_num`: Local designation number

To retrain the model:

```bash
pip install -e ".[train]"  # Install training dependencies
cd training
python train.py              # Train all stages
python train.py --stage 1    # Train only union detector
python train.py --stage 2    # Train only affiliation classifier
python train.py --stage 3    # Train only designation extractor
```

## Model Architecture

The model uses a three-stage contrastive extraction pipeline:

```
Input: "SEIU Local 1199"
         │
         ▼
┌─────────────────────────────┐
│  Tokenizer                  │
│  ["SEIU", " ", "Local", " ", "1199"]
│  token_type: [word, space, word, space, number]
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: Union Detection   │
│  (Contrastive)              │
│                             │
│  CharCNN → Projection →     │
│  Similarity to Union        │
│  Centroid                   │
│                             │
│  score=0.999 → is_union=True│
└─────────────────────────────┘
         │
         ▼ (if is_union)
┌─────────────────────────────┐
│  Stage 2: Affiliation       │
│  (Nearest Centroid)         │
│                             │
│  CharCNN → Projection →     │
│  Distance to Affiliation    │
│  Centroids                  │
│                             │
│  Nearest: SEIU (dist=0.009) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 3: Designation       │
│  (Pointer Network)          │
│                             │
│  CharCNN + Transformer +    │
│  BiLSTM + Affiliation       │
│  Embedding → Pointer Score  │
│                             │
│  Points to: "1199"          │
└─────────────────────────────┘
         │
         ▼
Output: {is_union: True, affiliation: "SEIU", designation: "1199"}
```

### Stage 1: Union Detection

Contrastive learning to distinguish union names from non-union text.

- **Encoder**: CharCNN + is_number embedding + projection head
- **Training**: One-class contrastive loss (union examples form positive pairs)
- **Inference**: Compute similarity to learned union centroid
- **Threshold**: Texts with similarity ≥ 0.5 classified as unions

### Stage 2: Affiliation Classification

Nearest-centroid classification in contrastive embedding space.

- **Encoder**: CharCNN + is_number embedding + projection head
- **Training**: Supervised contrastive loss (same-affiliation = positive pairs)
- **Inference**: Find nearest affiliation centroid
- **Threshold**: Distance > 0.20 → affiliation_unrecognized = True

### Stage 3: Designation Extraction

Pointer network that selects the correct local number.

- **Token Embeddings**: CharCNN (words) + special embedding (numbers, punct)
- **Context**: Transformer encoder (3 layers, 4 heads)
- **Selection**: BiLSTM + affiliation embedding → score each number token
- **Output**: Highest-scoring number token, or null if no designation

### Components

**Character CNN (for word tokens)**
- Character embedding: 16-dim
- Multi-scale 1D convolutions (kernel sizes 2, 3, 4, 5)
- Max pooling → 64-dim token embedding
- Typo-robust: handles misspellings gracefully

**Special Token Embedding (for non-words)**
- Lookup table for numbers, punctuation, spaces
- 64-dim embeddings

**Contrastive Projection**
- 2-layer MLP: 64+8 → 128 → 64
- L2 normalization for cosine similarity

### Model Statistics

- Parameters: ~3M total across all stages
- Inference: CPU or MPS (Apple Silicon)
- Model files: ~15MB total

### Performance

On labeled data (94,308 examples with known affiliations):

| Metric | All | Non-None Predictions |
|--------|-----|---------------------|
| Affiliation accuracy | 99.0% | 99.7% |
| Joint accuracy | 98.9% | 99.5% |

- Designation accuracy: 99.9%
- Only 0.7% of predictions return None (unrecognized affiliation)
