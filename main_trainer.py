import os
import sys
import argparse
import pathlib
import warnings
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from antigen_antibody_emb import antibody_antigen_dataset
from antibinder_model import AntiBinder, configuration, AntiModelInitial

warnings.filterwarnings("ignore")

# --- Patched torch.load for IgFold compatibility ---
try:
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        # This is a workaround for a breaking change in PyTorch 2.x affecting IgFold.
        # It forces pickle to be used, which is less secure but necessary for this case.
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
except AttributeError:
    print("Could not patch torch.load. This might cause issues with IgFold on newer PyTorch versions.")

# --- Logger Fallback ---
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
        # Set device for consistent tensor operations
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        if not load:
            self._init_model()

    def _init_model(self):
        try:
            init = AntiModelInitial()
            self.model.apply(init._init_weights)
            print("Model weights initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not apply custom weight initialization: {e}")

    def _calculate_metrics(self, predictions, targets):
        # Apply sigmoid since model now outputs raw logits
        probabilities = torch.sigmoid(predictions)
        pred_np = (probabilities.cpu().detach().numpy() > 0.5).astype(int)
        target_np = targets.cpu().numpy().astype(int)
        return {
            'accuracy': accuracy_score(target_np, pred_np),
            'precision': precision_score(target_np, pred_np, zero_division=0),
            'f1': f1_score(target_np, pred_np, zero_division=0),
            'recall': recall_score(target_np, pred_np, zero_division=0)
        }

    def _run_epoch(self, dataloader, criterion, optimizer=None):
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()
        
        total_loss = 0
        all_predictions, all_targets = [], []
        batch_count = 0

        for batch in tqdm(dataloader, desc="Training" if is_train else "Validation"):
            heavy_chain, light_chain, antigen, labels = batch['heavy_chain'], batch['light_chain'], batch['antigen'], batch['label']
            
            # Move labels to appropriate device
            labels = labels.to(self.device)

            with torch.set_grad_enabled(is_train):
                predictions = self.model(heavy_chain, light_chain, antigen)
                loss = criterion(predictions.squeeze(), labels.float())

                # Debug: Print some statistics for first few batches
                if batch_count < 3:
                    print(f"  Batch {batch_count}: pred range [{predictions.min().item():.3f}, {predictions.max().item():.3f}], "
                          f"loss: {loss.item():.3f}, labels: {labels.cpu().numpy()}")

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    # Add gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(labels.detach())
            batch_count += 1

        avg_loss = total_loss / len(dataloader)
        metrics = self._calculate_metrics(torch.cat(all_predictions), torch.cat(all_targets))
        return avg_loss, metrics

    def train(self, criterion, epochs, start_epoch=0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # Debug: Print initial model state
        print(f"Training with learning rate: {self.args.lr}")
        print(f"Criterion: {criterion}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, epochs):
            train_loss, train_metrics = self._run_epoch(self.train_dataloader, criterion, optimizer)
            val_loss, val_metrics = self._run_epoch(self.valid_dataloader, criterion)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

            self.logger.log([epoch + 1, train_loss, train_metrics['accuracy'], train_metrics['precision'], train_metrics['f1'], train_metrics['recall'],
                               val_loss, val_metrics['accuracy'], val_metrics['precision'], val_metrics['f1'], val_metrics['recall']])

            if val_loss < self.best_val_loss:
                print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                self.best_val_loss = val_loss
                self.save_model(epoch + 1, val_loss)

    def save_model(self, epoch, val_loss):
        filename = f"{self.args.model_name}_epoch{epoch}_valloss{val_loss:.4f}.pth"
        model_path = os.path.join(self.args.ckpt_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'args': self.args
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint and return the epoch to resume from"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Resumed from epoch {epoch}, best val loss: {self.best_val_loss:.4f}")
        return epoch

def custom_collate(batch):
    # This function will properly stack the tensors for each part of the data
    keys = ['heavy_chain', 'light_chain', 'antigen', 'label']
    res = {key: {sub_key: [] for sub_key in batch[0][key]} if isinstance(batch[0][key], dict) else [] for key in keys}

    for item in batch:
        for key in keys:
            if isinstance(item[key], dict):
                for sub_key in item[key]:
                    res[key][sub_key].append(item[key][sub_key])
            else:
                res[key].append(item[key])

    def pad_and_stack_tensors(tensor_list):
        """Pad tensors to the same size and stack them"""
        if not tensor_list or not isinstance(tensor_list[0], torch.Tensor):
            return torch.tensor(tensor_list) if tensor_list else torch.tensor([])
        
        # Find the maximum dimensions
        max_dims = [max(t.size(i) for t in tensor_list) for i in range(tensor_list[0].dim())]
        
        # Pad each tensor to match the maximum dimensions
        padded_tensors = []
        for tensor in tensor_list:
            pad_sizes = []
            for i in range(tensor.dim() - 1, -1, -1):  # padding is specified in reverse order
                pad_before = 0
                pad_after = max_dims[i] - tensor.size(i)
                pad_sizes.extend([pad_before, pad_after])
            
            if any(pad_sizes):
                padded_tensor = torch.nn.functional.pad(tensor, pad_sizes)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        
        return torch.stack(padded_tensors)

    for key in keys:
        if isinstance(res[key], dict):
            for sub_key in res[key]:
                res[key][sub_key] = pad_and_stack_tensors(res[key][sub_key])
        else:
            res[key] = pad_and_stack_tensors(res[key])
            
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AntiBinder model with heavy and light chains.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='AntiBinderV3')
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA training")
    parser.add_argument('--device', type=str, default='0', help="CUDA device ordinal")
    parser.add_argument('--data', type=str, default='train')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        # Enable CUDA debugging for better error messages
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # --- Setup Directories ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.log_dir = os.path.join(base_dir, 'logs')
    args.ckpt_dir = os.path.join(base_dir, 'ckpts')
    args.data_dir = os.path.join(base_dir, 'datasets')
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # --- Set Seeds ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Model and Config ---
    config = configuration()
    model = AntiBinder(config, latent_dim=args.latent_dim, res=True)
    if args.cuda:
        model = nn.DataParallel(model).cuda()
        print(f"Model loaded on {torch.cuda.device_count()} GPU(s)")
    else:
        print("Model loaded on CPU")

    # --- Datasets and Dataloaders ---
    train_path = os.path.join(args.data_dir, 'training_data.csv')
    test_path = os.path.join(args.data_dir, 'test_data.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Dataset files not found at {train_path} or {test_path}")
        sys.exit(1)

    train_dataset = antibody_antigen_dataset(antigen_config=config, antibody_config=config, data_path=train_path, train=True)
    test_dataset = antibody_antigen_dataset(antigen_config=config, antibody_config=config, data_path=test_path, test=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate)
    valid_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate)

    # --- Logger ---
    log_file = os.path.join(args.log_dir, f"{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    logger = CSVLogger_my(['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall', 'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall'], log_file)

    # --- Start Training ---
    trainer = Trainer(model, train_dataloader, valid_dataloader, args, logger)
    
    # Check if resuming from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Use BCEWithLogitsLoss - more numerically stable and works with raw logits
    criterion = nn.BCEWithLogitsLoss()
    trainer.train(criterion, args.epochs, start_epoch)
