import os
import sys
import argparse
import warnings
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, 
    recall_score, roc_auc_score, confusion_matrix
)

from antigen_antibody_emb import antibody_antigen_dataset
from antibinder_model import AntiBinder, configuration
from process_all_data import run_combined_chain_processing

warnings.filterwarnings("ignore")

# --- Patched torch.load for IgFold compatibility ---
try:
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
except AttributeError:
    print("Could not patch torch.load.")

class Tester:
    def __init__(self, model, dataloader, args):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        # Set device for consistent tensor operations
        self.device = torch.device('cuda' if args.cuda else 'cpu')

    def _calculate_metrics(self, y_true, y_pred, y_scores):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        y_scores = y_scores.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred).ravel()
        tn, fp, fn, tp = (cm[0], cm[1], cm[2], cm[3]) if len(cm) == 4 else (0, 0, 0, 0)

        return {
            'roc_auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }

    def predict_and_save(self):
        self.model.eval()
        all_preds, all_targets, all_scores, all_indices = [], [], [], []

        with torch.no_grad():
            for batch in self.dataloader:
                heavy_chain, light_chain, antigen, labels, indices = \
                    batch['heavy_chain'], batch['light_chain'], batch['antigen'], batch['label'], batch['index']

                # Move labels to appropriate device
                labels = labels.to(self.device)

                scores = self.model(heavy_chain, light_chain, antigen).squeeze()
                # Apply sigmoid to convert logits to probabilities (0-1 range)
                probabilities = torch.sigmoid(scores)
                preds = (probabilities > 0.5).long()

                all_preds.append(preds)
                all_targets.append(labels)
                all_scores.append(probabilities)  # Store probabilities instead of raw scores
                all_indices.append(indices)

        # Concatenate all batch results
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_scores = torch.cat(all_scores)
        all_indices = torch.cat(all_indices).cpu().numpy()

        # Save results to CSV
        results_df = self.dataloader.dataset.data.iloc[all_indices].copy()
        results_df['predicted_probability'] = all_scores.cpu().numpy()
        results_df['predicted_label'] = all_preds.cpu().numpy()
        
        output_dir = "predictions/output"
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, os.path.basename(self.args.input_path).replace('.csv', '_results.csv'))
        results_df.to_csv(result_path, index=False)
        print(f"Results saved to {result_path}")

        # Calculate and print metrics
        metrics = self._calculate_metrics(all_targets, all_preds, all_scores)
        print("--- Test Metrics ---")
        for key, value in metrics.items():
            print(f"{key.capitalize()}: {value:.4f}")
        return metrics

def custom_collate(batch):
    keys = ['heavy_chain', 'light_chain', 'antigen', 'label', 'index']
    res = {key: {sub_key: [] for sub_key in batch[0][key]} if isinstance(batch[0][key], dict) else [] for key in keys if key in batch[0]}

    for item in batch:
        for key in keys:
            if key not in item: continue
            if isinstance(item[key], dict):
                for sub_key in item[key]:
                    res[key][sub_key].append(item[key][sub_key])
            else:
                res[key].append(item[key])

    for key in keys:
        if key not in res: continue
        if isinstance(res[key], dict):
            for sub_key in res[key]:
                # Handle variable-length tensors by padding
                if len(res[key][sub_key]) > 0 and isinstance(res[key][sub_key][0], torch.Tensor):
                    # Find max length in this batch
                    max_len = max(tensor.size(0) for tensor in res[key][sub_key])
                    # Pad all tensors to max length
                    padded_tensors = []
                    for tensor in res[key][sub_key]:
                        if tensor.size(0) < max_len:
                            pad_size = max_len - tensor.size(0)
                            padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_size))
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    res[key][sub_key] = torch.stack(padded_tensors)
                else:
                    res[key][sub_key] = torch.stack(res[key][sub_key])
        else:
            res[key] = torch.stack(res[key]) if isinstance(res[key][0], torch.Tensor) else torch.tensor(res[key])
            
    return res

def check_data_needs_splitting(df):
    """
    Check if the dataframe needs to be split into CDR/FR regions.
    Returns True if data needs splitting (only has vh, vl, Antigen Sequence)
    Returns False if data already has CDR/FR regions
    """
    required_cdr_fr_columns = [
        'H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4',
        'L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4'
    ]
    
    # Check if all CDR/FR columns are present
    has_cdr_fr_regions = all(col in df.columns for col in required_cdr_fr_columns)
    
    # Check if basic columns are present
    has_basic_columns = all(col in df.columns for col in ['vh', 'vl', 'Antigen Sequence'])
    
    if has_cdr_fr_regions and has_basic_columns:
        print("✅ Data already contains CDR/FR regions - proceeding with prediction")
        return False
    elif has_basic_columns and not has_cdr_fr_regions:
        print("🔄 Data contains vh/vl sequences but no CDR/FR regions - will automatically split")
        return True
    elif has_cdr_fr_regions and not has_basic_columns:
        print("⚠️  Data has CDR/FR regions but missing basic vh/vl columns - this may cause issues")
        return False
    else:
        missing_cols = []
        if 'vh' not in df.columns:
            missing_cols.append('vh')
        if 'vl' not in df.columns:
            missing_cols.append('vl')
        if 'Antigen Sequence' not in df.columns:
            missing_cols.append('Antigen Sequence')
        
        # If we have some CDR/FR columns but not all, list what's missing
        missing_cdr_fr = [col for col in required_cdr_fr_columns if col not in df.columns]
        if missing_cdr_fr and len(missing_cdr_fr) < len(required_cdr_fr_columns):
            print(f"⚠️  Partial CDR/FR data detected. Missing CDR/FR columns: {missing_cdr_fr}")
            print("    Will attempt to process from vh/vl sequences if available...")
            if has_basic_columns:
                return True
        
        raise ValueError(f"Input data is missing required columns: {missing_cols + missing_cdr_fr}")

def process_antibody_data(input_path, scheme="chothia"):
    """
    Process antibody data by splitting into CDR/FR regions if needed.
    Returns the path to the processed data (either original or newly created).
    """
    print(f"📂 Loading input data from: {input_path}")
    
    # Load the input data to check its structure
    try:
        df = pd.read_csv(input_path)
        print(f"📊 Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading input file: {str(e)}")
        sys.exit(1)
    
    # Check if data needs splitting
    if not check_data_needs_splitting(df):
        # Data already has CDR/FR regions, return original path
        print(f"📄 Using original data file: {input_path}")
        return input_path
    
    # Data needs splitting - create processed version
    print(f"🔧 Processing {len(df)} antibody sequences using {scheme} numbering scheme...")
    
    # Create temporary processed file path
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_dir = os.path.join(os.path.dirname(input_path), 'temp_processed')
    os.makedirs(temp_dir, exist_ok=True)
    processed_path = os.path.join(temp_dir, f"{base_name}_processed.csv")
    
    try:
        # Use the existing professional processing function
        run_combined_chain_processing(df, processed_path, scheme)
        
        print(f"✅ Successfully processed antibody sequences!")
        print(f"📄 Processed data saved to: {processed_path}")
        print("   Heavy chain regions: H-FR1, H-CDR1, H-FR2, H-CDR2, H-FR3, H-CDR3, H-FR4")
        print("   Light chain regions: L-FR1, L-CDR1, L-FR2, L-CDR2, L-FR3, L-CDR3, L-FR4")
        
        return processed_path
        
    except Exception as e:
        print(f"❌ Error during antibody sequence processing: {str(e)}")
        print("\nCommon issues:")
        print("- Make sure your CSV has 'vh', 'vl', and 'Antigen Sequence' columns")
        print("- Ensure ANARCI/abnumber is properly installed")
        print("- Check that the antibody sequences are valid protein sequences")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test AntiBinder model.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        sys.exit(1)

    # --- Model and Config ---
    config = configuration()
    model = AntiBinder(config, latent_dim=args.latent_dim, res=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cuda' if args.cuda else 'cpu')
    # Adjust for DataParallel wrapper
    state_dict = checkpoint['model_state_dict']
    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    if args.cuda:
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
    else:
        print("Model loaded on CPU")

    # --- Process input data (split into CDR/FR regions if needed) ---
    print("🔍 Checking input data format...")
    processed_data_path = process_antibody_data(args.input_path)
    
    # --- Dataset and Dataloader ---
    dataset = antibody_antigen_dataset(antigen_config=config, antibody_config=config, data_path=processed_data_path, test=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate)

    # --- Run Testing ---
    tester = Tester(model=model, dataloader=dataloader, args=args)
    tester.predict_and_save()
