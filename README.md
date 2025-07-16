# AntiBinder3
AntiBinder3: A sequence-structure hybrid model based on bidirectional cross-attention mechanism for predicting antibody-antigen binding relationships. This version supports both **Heavy Chain (VH)** and **Light Chain (VL)** data for enhanced prediction accuracy.

![framework](./figures/model_all.png)

## Introduction
This project predicts antigen-antibody binding affinity for protein sequences. The model can be trained and used based on sequence data, leveraging both VH and VL regions of antibodies for comprehensive binding prediction. The model combines:
- **Sequence embeddings** using ESM-2 for antigens and tokenized sequences for antibodies
- **Structure embeddings** using IgFold for antibody chains
- **Bidirectional cross-attention** between antibody and antigen representations

## Key Features
- ✅ **Dual-chain antibody modeling** (Heavy + Light chains)
- ✅ **Professional antibody region splitting** using ANARCI/abnumber
- ✅ **Structure-aware embeddings** with IgFold
- ✅ **Efficient caching** with LMDB for embeddings
- ✅ **GPU acceleration** with CUDA support
- ✅ **Proper probability outputs** with sigmoid activation

## Dependencies
- Python 3.11
- PyTorch (with CUDA support recommended)
- ESM-2 (protein language model)
- IgFold (antibody structure prediction)
- ANARCI/abnumber (antibody numbering)

## Installation Guide

### Clone the repository
```bash
git clone https://github.com/brisingr10/Antibinder3.git
cd Antibinder3
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Data Preparation
Your raw antibody-antigen binding data needs to be processed to extract CDR/FR regions and prepare for training.

#### Option A: Use Existing Processing Pipeline
For standard datasets (CoV-AbDab, BioMap, etc.):

```bash
python process_all_data.py
```

This script:
- Reads raw data from `datasets/raw_data/`
- Splits antibody sequences into CDR/FR regions using professional tools
- Combines all processed data into training/test sets
- Outputs: `datasets/training_data.csv` and `datasets/test_data.csv`

#### Option B: Process Your Own Data
For custom datasets with `vh`, `vl`, and `Antigen Sequence` columns:

```bash
python use_existing_splitter.py
```

Modify the script to point to your input file. This will split the antibody sequences into proper CDR/FR regions.
    `ANT_Binding`

### Data Structure Details

The processed data consists of antibody-antigen binding pairs with the following structure:

**Required Columns:**
- `vh`: Heavy chain sequence
- `vl`: Light chain sequence  
- `Antigen Sequence`: Target protein sequence
- `H-FR1`, `H-CDR1`, `H-FR2`, `H-CDR2`, `H-FR3`, `H-CDR3`, `H-FR4`: Heavy chain regions
- `L-FR1`, `L-CDR1`, `L-FR2`, `L-CDR2`, `L-FR3`, `L-CDR3`, `L-FR4`: Light chain regions
- `ANT_Binding`: Binary binding label (0 = no binding, 1 = binding)

**Antibody Regions:**
- **Framework Regions (FR1-4)**: Structural scaffold regions
- **Complementarity Determining Regions (CDR1-3)**: Antigen-binding regions
- **CDR3**: Most variable and critical for binding specificity

**Data Sources Supported:**
- CoV-AbDab: COVID-19 antibody database
- BioMap: Binding affinity measurements
- Custom datasets following the same format

### 2. Training the Model
Train the AntiBinder model using your processed data:

```bash
python main_trainer.py \
    --batch_size 32 \
    --latent_dim 32 \
    --epochs 50 \
    --lr 1e-4 \
    --model_name AntiBinderV3 \
    --device 0

# Optional arguments:
# --no_cuda: Use CPU instead of GPU
# --seed: Random seed for reproducibility (default: 42)
```

**Training Features:**
- Automatic model checkpointing
- Validation loss monitoring
- CUDA acceleration
- Mixed precision training support

### 3. Making Predictions
Use a trained model to predict antibody-antigen binding:

```bash
python main_test.py \
    --input_path "predictions/your_data.csv" \
    --checkpoint_path "ckpts/AntiBinderV3_epoch31_valloss0.4168.pth" \
    --batch_size 64

# Optional arguments:
# --no_cuda: Use CPU instead of GPU
# --latent_dim: Model latent dimension (default: 32)
# --seed: Random seed (default: 42)
```

**Input Requirements:**
- CSV file with `vh`, `vl`, and `Antigen Sequence` columns
- CDR/FR regions will be automatically extracted if not present
- No binding labels required for prediction

**Output:**
- Results saved to `predictions/output/[filename]_results.csv`
- Includes `predicted_probability` (0-1) and `predicted_label` (0/1)
- Performance metrics displayed if ground truth available

### 4. Processing New Antibody Data
To split new antibody sequences into CDR/FR regions:

```bash
python use_existing_splitter.py
```

Edit the script to specify your input file. This uses the same professional antibody annotation tools (ANARCI/abnumber) as the training pipeline.

## Model Architecture
The model architecture (`antibinder_model.py`) consists of:

**Core Components:**
- `Combine_Embedding`: Integrates sequence and structure embeddings for antibody chains
- `BiCrossAttentionBlock`: Bidirectional cross-attention between antibody and antigen
- `AntiBinder`: Main model orchestrating all components

**Key Features:**
- **Multi-modal embeddings**: Combines sequence tokens with structure information
- **Attention mechanisms**: Bidirectional cross-attention for antibody-antigen interaction
- **Residual connections**: Enhanced gradient flow and training stability
- **Proper output activation**: Sigmoid for probability outputs (0-1 range)

## File Structure
```
AntiBinder3/
├── main_trainer.py          # Training script
├── main_test.py             # Prediction script  
├── process_all_data.py      # Data preprocessing pipeline
├── use_existing_splitter.py # Custom data processing
├── antibinder_model.py      # Model architecture
├── antigen_antibody_emb.py  # Data loading and embedding
├── cfg_ab.py               # Configuration parameters
├── datasets/               # Training and test data
├── ckpts/                  # Model checkpoints
├── predictions/            # Prediction inputs/outputs
└── figures/                # Architecture diagrams
```

## Performance
The model achieves competitive performance on antibody-antigen binding prediction:
- Trained on combined datasets (CoV-AbDab, BioMap)
- Validates on held-out test sets
- Metrics: ROC-AUC, Precision, Recall, F1-score
- Proper probability calibration with sigmoid outputs

## Cache Management
Embeddings are cached using LMDB for efficiency:
- **Antigen embeddings**: `antigen_esm/` (ESM-2 representations)
- **Antibody structures**: `datasets/fold_emb/` (IgFold embeddings)
- **Automatic caching**: First run processes, subsequent runs load from cache

## Troubleshooting

### Common Issues
1. **Missing CDR/FR regions**: Use `use_existing_splitter.py` to process sequences
2. **CUDA out of memory**: Reduce `batch_size` or use `--no_cuda`
3. **Probability > 1 or < 0**: Fixed in latest version with proper sigmoid activation
4. **Missing dependencies**: Ensure all packages in `requirements.txt` are installed

### GPU Requirements
- Recommended: NVIDIA GPU with 8GB+ VRAM
- CPU training supported but slower
- Mixed precision training for memory efficiency

## Citation
If you use AntiBinder3 in your research, please cite:
```
[Citation information to be added]
```

## License
[License information to be added]