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
                # Convert logits to probabilities using sigmoid
                probabilities = torch.sigmoid(scores)
                preds = (probabilities > 0.5).long()

                all_preds.append(preds)
                all_targets.append(labels)
                all_scores.append(probabilities)
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
                if sub_key == 'structure':
                    # Handle variable-length structure embeddings by padding
                    max_len = max(tensor.shape[0] for tensor in res[key][sub_key])
                    padded_tensors = []
                    for tensor in res[key][sub_key]:
                        if tensor.shape[0] < max_len:
                            padding = torch.zeros(max_len - tensor.shape[0], tensor.shape[1])
                            padded_tensor = torch.cat([tensor, padding], dim=0)
                            padded_tensors.append(padded_tensor)
                        else:
                            padded_tensors.append(tensor)
                    res[key][sub_key] = torch.stack(padded_tensors)
                else:
                    res[key][sub_key] = torch.stack(res[key][sub_key])
        else:
            res[key] = torch.stack(res[key]) if isinstance(res[key][0], torch.Tensor) else torch.tensor(res[key])
            
    return res

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

    # --- Dataset and Dataloader ---
    dataset = antibody_antigen_dataset(antigen_config=config, antibody_config=config, data_path=args.input_path, test=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate)

    # --- Run Testing ---
    tester = Tester(model=model, dataloader=dataloader, args=args)
    tester.predict_and_save()
