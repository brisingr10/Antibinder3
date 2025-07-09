# AntiBinder
AntiBinder: a sequence-structure hybrid model based on bidirectional cross-attention mechanism for predicting antibody-antigen binding relationships. This version now supports both **Heavy Chain (VH)** and **Light Chain (VL)** data for enhanced prediction accuracy.

![framework](./figures/model_all.png)

## Introduction
This project is used for predicting antigen-antibody affinity for protein types. The model can be trained and used based solely on sequence data. You can also stack the modules within, making the model's parameters significantly larger, and train it to achieve a plug-and-play effect. The updated model now leverages both VH and VL regions of antibodies for more comprehensive binding prediction.

## Dependencies
python 3.11

## Installation Guide
Detailed instructions on how to install and set up the project:

### Clone the repository
git clone https://github.com/brisingr10/Antibinder3.git

### Install dependencies
pip install -r requirements.txt

## Usage Instructions

### 1. Data Preparation
Before training or prediction, your raw antibody-antigen binding data needs to be processed to extract relevant regions and combine them into a single dataset.

**Steps:**

1.  **Process Raw Data:**
    Use `process_all_data.py` to perform all necessary data preprocessing, including:
    *   Reading raw data from `datasets/raw_data/`.
    *   Renaming columns (e.g., `Heavy` to `vh`, `Light` to `vl`, `antigen` to `Antigen Sequence`, `Label` to `ANT_Binding`) based on internal configuration for known datasets.
    *   Splitting Heavy Chain (VH) and Light Chain (VL) sequences into their respective Framework (FR) and Complementarity Determining Region (CDR) segments (H-FR1, H-CDR1, etc., and L-FR1, L-CDR1, etc.).
    *   Combining all processed data files into a single dataset.
    *   Performing data validation (dropping rows with missing VH or VL sequences and duplicates).
    *   Splitting the combined data into `combined_training_data.csv` and `test_data.csv` for training and validation.

    Ensure your raw data CSV files in `datasets/raw_data/` contain the necessary columns for heavy chain, light chain, antigen sequence, and binding labels, as expected by `process_all_data.py` (refer to the script's `if __name__ == "__main__":` block for specific column mappings for each dataset).

    ```bash
    python process_all_data.py
    # This script reads from datasets/raw_data/, outputs intermediate processed files to datasets/process_data/,
    # and saves the final combined_training_data.csv and test_data.csv to the datasets/ directory.
    ```

    **Output Data Structure:**
    The `combined_training_data.csv` and `test_data.csv` files will contain the following columns:
    `vh`, `vl`, `Antigen Sequence`,
    `H-FR1`, `H-CDR1`, `H-FR2`, `H-CDR2`, `H-FR3`, `H-CDR3`, `H-FR4`,
    `L-FR1`, `L-CDR1`, `L-FR2`, `L-CDR2`, `L-FR3`, `L-CDR3`, `L-FR4`,
    `ANT_Binding`

### 2. Training the Model
Once your data is prepared and combined, you can train the AntiBinder model using `main_trainer.py`.

```bash
python main_trainer.py \
    --batch_size 32 \
    --latent_dim 32 \
    --epochs 50 \
    --lr 1e-4 \
    --model_name AntiBinderV2 \
    --device 0 # Specify CUDA device if available, e.g., 0, 1, etc.

# Other optional arguments:
# --no_cuda: Disable CUDA training (use CPU)
# --seed: Random seed for reproducibility (default: 42)
```

*   **Configuration:** Model parameters and architecture configurations (e.g., `max_position_embeddings` for heavy and light chains, region type indexing) are defined in `cfg_ab.py`.
*   **Data Loading & Embedding:** The `antigen_antibody_emb.py` script handles loading the combined data, generating ESM embeddings for antigens, and IgFold structure embeddings for antibody chains. It also manages LMDB caches for efficient data retrieval.

### 3. Predicting with the Model
To make predictions using a trained model, use `main_test.py`.

```bash
python main_test.py \
    --input_path "path/to/your/prediction_data.csv" \
    --checkpoint_path "path/to/your/trained_model.pth" \
    --batch_size 64

# Other optional arguments:
# --no_cuda: Disable CUDA (use CPU)
# --seed: Random seed for reproducibility (default: 42)
```

*   **Input Data for Prediction:** The `--input_path` should point to a CSV file containing `vh`, `vl`, and `Antigen Sequence` columns, processed in the same way as the training data (i.e., split into FR/CDR regions using `heavy_chain_split.py` and `light_chain_split.py`). The `ANT_Binding` column is not strictly required for prediction, but the script expects the same data structure.
*   **Output:** The script will generate a new CSV file in the `predictions/output/` directory, appending `_results.csv` to the input filename. This file will include `predicted_probability` and `predicted_label` columns.

## Model Architecture
The core model architecture is defined in `antibinder_model.py`, which includes:
*   `Combine_Embedding`: Handles the combination of sequence and structure embeddings for both heavy and light chains.
*   `BiCrossAttentionBlock`: Implements the bidirectional cross-attention mechanism between antibody (combined VH+VL) and antigen embeddings.
*   `AntiBinder`: The main model class that orchestrates the embedding combination, cross-attention, and final classification layers.

## Cache Management
ESM and IgFold embeddings are cached using LMDB for faster subsequent runs. Separate cache directories are maintained for heavy chain structures, light chain structures, and antigen ESM embeddings within the `datasets/fold_emb/` and `antigen_esm/` directories respectively.