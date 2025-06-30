import os
import sys
import argparse
import warnings
import pandas as pd
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from sklearn.metrics import (accuracy_score, precision_score, f1_score, 
                           recall_score, roc_auc_score, confusion_matrix)

# Local imports
from antigen_antibody_emb import * 
from antibinder_model import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Trainer:
    def __init__(self, model, valid_dataloader, args):
        self.model = model
        self.valid_dataloader = valid_dataloader
        self.args = args

    def _calculate_metrics(self, yhat, y, yscores):
        """Calculate evaluation metrics"""
        cm = confusion_matrix(y, yhat).ravel()
        TN, FP, FN, TP = 0, 0, 0, 0
        
        if len(cm) == 1:
            if y[0].item() == yhat[0].item() == 0:
                TN = cm[0]
            elif y[0].item() == yhat[0].item() == 1:
                TP = cm[0]
        else:
            TN, FP, FN, TP = cm
        
        roc_auc = roc_auc_score(y, yscores) if len(np.unique(y)) > 1 else None
        
        return (roc_auc, precision_score(y, yhat), accuracy_score(y, yhat), 
                recall_score(y, yhat), f1_score(y, yhat), TN, FP, FN, TP)
    
    def predict_and_save(self):
        """Run predictions and save results"""
        self.model.eval()
        Y_hat = []
        Y = []
        Y_scores = []
        all_probs = []
        all_indices = []
        all_rows = []
        
        with torch.no_grad():
            for batch_idx, (antibody_set, antigen_set, label, indices) in enumerate(self.valid_dataloader):
                probs = self.model(antibody_set, antigen_set)
                y = label.float()
                yhat = (probs > 0.5).long().cuda()
                y_scores = probs

                for i, prob in enumerate(probs):
                    all_probs.append(prob.item())
                    all_indices.append(indices[i].item())

                Y_hat.extend(yhat)
                Y.extend(y)
                Y_scores.extend(y_scores)

                batch_rows = self.valid_dataloader.dataset.data.iloc[indices.cpu().numpy()]
                all_rows.append(batch_rows)

        self._save_results(all_rows, all_probs)
        
        auc, precision, accuracy, recall, f1, TN, FP, FN, TP = self._calculate_metrics(
            (torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
            torch.tensor(Y),
            (torch.cat([temp2.view(1, -1) for temp2 in Y_scores], dim=0)).cpu().numpy()
        )
        
        return auc, precision, accuracy, recall, f1, TN, FP, FN, TP
    
    def _save_results(self, all_rows, all_probs):
        """Save predictions to CSV file"""
        all_rows_df = pd.concat(all_rows, axis=0).reset_index(drop=True)
        all_rows_df['predicted_probability'] = all_probs
        
        result_filename = os.path.splitext(os.path.basename(self.args.input_path))[0] + "_result.csv"
        output_dir = "predictions/output"
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, result_filename)
        
        all_rows_df.to_csv(result_path, index=False)

def main():
    """Main function to run antibody-antigen binding prediction"""
    parser = argparse.ArgumentParser(description='AntiBinder Testing Script')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--model_name', type=str, default='AntiBinder')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, 
                        default='ckpts/AntiBinder_train_128_40_36_6e-05.pth')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.input_path):
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        sys.exit(1)

    # Model configuration
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)
    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)
    
    # Initialize model
    model = antibinder(
        antibody_hidden_dim=1024, 
        antigen_hidden_dim=1024, 
        latent_dim=args.latent_dim, 
        res=False
    ).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Create dataset and dataloader
    dataset = antibody_antigen_dataset(
        antigen_config=antigen_config,
        antibody_config=antibody_config,
        data_path=args.input_path,
        train=False,
        test=True,
        rate1=0
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)

    # Run predictions
    trainer = Trainer(model=model, valid_dataloader=dataloader, args=args)
    trainer.predict_and_save()


if __name__ == "__main__":
    main()