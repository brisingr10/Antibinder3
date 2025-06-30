import os
import sys
import argparse
import pathlib
import warnings
from antigen_antibody_emb import * 
from antibinder_model import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

warnings.filterwarnings("ignore")

# Fix for PyTorch 2.6+ compatibility with IgFold
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Try to import custom logger, use fallback if not available
try:
    from utils.utils import CSVLogger_my
except ImportError:
    import csv
    class CSVLogger_my:
        def __init__(self, fieldnames, filename):
            self.fieldnames = fieldnames
            self.filename = filename
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                
        def log(self, values):
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(values)


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, args, logger, load=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        self.best_val_loss = float('inf')

        if not load:
            self._init_model()

    def _init_model(self):
        try:
            init = AntiModelIinitial()
            self.model.apply(init._init_weights)
            print("Model initialized successfully!")
        except Exception as e:
            print(f"Error during model initialization: {e}")
            print("Continuing with default initialization")

    def _calculate_metrics(self, predictions, targets):
        pred_np = predictions.long().cpu().numpy()
        target_np = targets.cpu().numpy()
        return (accuracy_score(target_np, pred_np), 
                precision_score(target_np, pred_np, zero_division=0), 
                f1_score(target_np, pred_np, zero_division=0), 
                recall_score(target_np, pred_np, zero_division=0))

    def validate(self, criterion):
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in self.valid_dataloader:
                if len(batch_data) != 3:
                    continue
                    
                antibody_set, antigen_set, labels = batch_data
                if self.args.cuda and torch.cuda.is_available():
                    labels = labels.cuda()
                    
                predictions = self.model(antibody_set, antigen_set)
                binary_predictions = (predictions > 0.5).long()
                loss = criterion(predictions.view(-1), labels.float().view(-1))
                
                val_loss += loss.item()
                all_predictions.extend(binary_predictions.cpu())
                all_targets.extend(labels.cpu())
        
        val_loss /= len(self.valid_dataloader)
        pred_tensor = torch.cat([p.view(1, -1) for p in all_predictions], dim=0)
        target_tensor = torch.tensor(all_targets)
        val_acc, val_precision, val_f1, val_recall = self._calculate_metrics(pred_tensor, target_tensor)
        
        return val_loss, val_acc, val_precision, val_f1, val_recall
    
    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            all_predictions = []
            all_targets = []
            
            for batch_data in tqdm(self.train_dataloader):
                if len(batch_data) != 3:
                    continue
                    
                antibody_set, antigen_set, labels = batch_data
                if self.args.cuda and torch.cuda.is_available():
                    labels = labels.cuda()
                    
                predictions = self.model(antibody_set, antigen_set)
                binary_predictions = (predictions > 0.5).long()
                loss = criterion(predictions.view(-1), labels.float().view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                all_predictions.extend(binary_predictions.cpu())
                all_targets.extend(labels.cpu())
            
            train_loss /= len(self.train_dataloader)
            pred_tensor = torch.cat([p.view(1, -1) for p in all_predictions], dim=0)
            target_tensor = torch.tensor(all_targets)
            train_acc, train_precision, train_f1, train_recall = self._calculate_metrics(pred_tensor, target_tensor)
            
            # Validation
            val_loss, val_acc, val_precision, val_f1, val_recall = self.validate(criterion)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            self.logger.log([epoch+1, train_loss, train_acc, train_precision, train_f1, train_recall,
                           val_loss, val_acc, val_precision, val_f1, val_recall])
            
            # Save best model
            if val_loss < self.best_val_loss:
                print(f'Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}, saving model...')
                self.best_val_loss = val_loss
                self.save_model(epoch+1, val_loss)   
    
    def save_model(self, epoch=None, val_loss=None):
        if epoch is not None and val_loss is not None:
            filename = f"{self.args.model_name}_{self.args.data}_best_epoch{epoch}_valloss{val_loss:.4f}.pth"
        else:
            filename = f"{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth"
        
        model_path = os.path.join(self.args.ckpt_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'args': self.args
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='AntiBinder')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--data', type=str, default='train')
    parser.add_argument('--base_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    
    # CUDA setup
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Setup directories
    args.base_dir = os.path.abspath(args.base_dir)
    args.log_dir = os.path.join(args.base_dir, 'logs')
    args.ckpt_dir = os.path.join(args.base_dir, 'ckpts')
    args.data_dir = os.path.join(args.base_dir, 'datasets')
    
    for directory in [args.log_dir, args.ckpt_dir, args.data_dir]:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Model setup
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)
    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)
    
    model = antibinder(antibody_hidden_dim=1024, antigen_hidden_dim=1024, latent_dim=args.latent_dim, res=False)
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    
    # Dataset paths
    train_path = os.path.join(args.data_dir, 'combined_training_data.csv')
    test_path = os.path.join(args.data_dir, 'test_data.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Dataset files not found")
        sys.exit(1)
    
    # Create datasets
    train_dataset = antibody_antigen_dataset(
        antigen_config=antigen_config, antibody_config=antibody_config,
        data_path=train_path, train=True, test=False, rate1=1
    )
    test_dataset = antibody_antigen_dataset(
        antigen_config=antigen_config, antibody_config=antibody_config,  
        data_path=test_path, train=False, test=True, rate1=1
    )
    
    # Custom collate function
    def custom_collate(batch):
        if len(batch[0]) == 4:
            antibody_sets, antigen_sets, labels, _ = zip(*batch)
        else:
            antibody_sets, antigen_sets, labels = zip(*batch)
        
        antibody_sets = [torch.stack(x) for x in zip(*antibody_sets)]
        antigen_sets = [torch.stack(x) for x in zip(*antigen_sets)]
        labels = torch.stack(labels)
        return antibody_sets, antigen_sets, labels
    
    # Data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate)
    valid_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate)
    
    # Logger
    log_file = os.path.join(args.log_dir, f"{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    logger = CSVLogger_my(
        ['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall',
         'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall'],
        log_file
    )
    
    # Start training
    trainer = Trainer(model, train_dataloader, valid_dataloader, args, logger)
    criterion = nn.BCELoss()
    trainer.train(criterion, args.epochs)