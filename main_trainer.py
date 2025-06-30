import os
from antigen_antibody_emb import * 
from antibinder_model import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
import sys 
import argparse
import pathlib
import warnings 
warnings.filterwarnings("ignore")

# Fix for PyTorch 2.6+ compatibility with IgFold
import torch
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Patched torch.load to handle IgFold compatibility issues"""
    # Always use weights_only=False for compatibility with IgFold and other libraries
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

print("Applied PyTorch compatibility patch for IgFold")

# Ensure utils module exists or provide fallback
try:
    from utils.utils import CSVLogger_my
except ImportError:
    print("Warning: CSVLogger_my not found, using fallback implementation")
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

from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Trainer():
    def __init__(self,model,train_dataloader, valid_dataloader, args,logger,load=False) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # self.test_dataloader = test_dataloader
        self.args = args
        self.logger = logger
        self.best_val_loss = float('inf')  # Track best validation loss
        self.load = load

        if self.load==False:
            self.init()
        else:
            print("no init model")

    def init(self):
        try:
            init = AntiModelIinitial()
            self.model.apply(init._init_weights)
            print("init successfully!")
        except Exception as e:
            print(f"Error during model initialization: {e}")
            print("Continuing with default initialization")

    def matrix(self,yhat,y):
        return sum(y==yhat)
    
    def matrix_val(self,yhat,y):
        return accuracy_score(y,yhat), precision_score(y, yhat), f1_score(y,yhat), recall_score(y, yhat)

    def validate(self, criterion):
        self.model.eval()
        val_loss = 0
        Y_hat = []
        Y = []
        with torch.no_grad():
            for batch_data in self.valid_dataloader:
                try:
                    # Handle different data formats
                    if len(batch_data) == 3:
                        antibody_set, antigen_set, label = batch_data
                    else:
                        print(f"Unexpected batch format with {len(batch_data)} elements: {[type(x) for x in batch_data]}")
                        continue
                        
                    probs = self.model(antibody_set, antigen_set)
                    yhat = (probs > 0.5).long()
                    y = label.float()
                    if self.args.cuda and torch.cuda.is_available():
                        y = y.cuda()
                    loss = criterion(probs.view(-1), y.view(-1))
                    val_loss += loss.item()
                    Y_hat.extend(yhat)
                    Y.extend(y)
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        val_loss = val_loss / len(self.valid_dataloader)
        val_acc, val_precision, val_f1, val_recall = self.matrix_val(
            (torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
            torch.tensor(Y).cpu()
        )
        return val_loss, val_acc, val_precision, val_f1, val_recall
    
    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        for epoch in range(epochs):
            self.model.train(True)
            train_acc = 0
            train_loss = 0
            num_train = 0
            Y_hat = []
            Y = []
            try:
                for batch_data in tqdm(self.train_dataloader):
                    try:
                        # Handle different data formats
                        if len(batch_data) == 3:
                            antibody_set, antigen_set, label = batch_data
                        else:
                            print(f"Unexpected batch format with {len(batch_data)} elements: {[type(x) for x in batch_data]}")
                            continue
                            
                        probs = self.model(antibody_set, antigen_set)
                        yhat = (probs > 0.5).long()
                        y = label.float()
                        if self.args.cuda and torch.cuda.is_available():
                            y = y.cuda()
                        loss = criterion(probs.view(-1), y.view(-1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        num_train += antibody_set[0].shape[0]
                        Y_hat.extend(yhat)
                        Y.extend(y)
                    except Exception as batch_e:
                        print(f"Error in training batch: {batch_e}")
                        continue
                train_acc, train_precision, train_f1, recall = self.matrix_val(
                    (torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                    torch.tensor(Y).cpu()
                )
                train_loss = train_loss / len(self.train_dataloader)

                # --- Validation step ---
                val_loss, val_acc, val_precision, val_f1, val_recall = self.validate(criterion)
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

                self.logger.log([
                    epoch+1, train_loss, train_acc, train_precision, train_f1, recall,
                    val_loss, val_acc, val_precision, val_f1, val_recall
                ])

                # Save checkpoint if validation loss improved
                if val_loss < self.best_val_loss:
                    print(f'Epoch {epoch+1}: Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}, saving model...')
                    self.best_val_loss = val_loss
                    self.save_model(epoch+1, val_loss)
            except Exception as e:
                print(f"Error during training epoch {epoch}: {e}")
                continue   
    
    def save_model(self, epoch=None, val_loss=None):
        try:
            # Create filename with epoch and validation loss info for best checkpoints
            if epoch is not None and val_loss is not None:
                model_path = os.path.join(self.args.ckpt_dir, 
                    f"{self.args.model_name}_{self.args.data}_best_epoch{epoch}_valloss{val_loss:.4f}.pth")
            else:
                # Fallback to original naming if no epoch/loss provided
                model_path = os.path.join(self.args.ckpt_dir, 
                    f"{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth")
            
            # Save model state dict along with additional info
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'args': self.args
            }
            torch.save(checkpoint, model_path)
            print(f"Best model checkpoint saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=36)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default='AntiBinder')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--data', type=str, default='train')
    parser.add_argument('--lmdb_size_gb', type=int, default=50, help='LMDB map size in GB')
    parser.add_argument('--base_dir', type=str, 
                    default=os.path.dirname(os.path.abspath(__file__)), 
                    help='Base directory for all data and output files')
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.cuda and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        args.cuda = False
    
    # Set CUDA device after checking availability
    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print(f"Using CUDA device: {args.device}")
    else:
        print("Using CPU for computation")

    # Create necessary directories
    args.base_dir = os.path.abspath(args.base_dir)
    args.log_dir = os.path.join(args.base_dir, 'logs')
    args.ckpt_dir = os.path.join(args.base_dir, 'ckpts')
    args.data_dir = os.path.join(args.base_dir, 'datasets')
    args.antigen_esm_dir = os.path.join(args.base_dir, 'antigen_esm', 'train')
    args.fold_emb_dir = os.path.join(args.base_dir, 'datasets', 'fold_emb', 'fold_emb_for_train')

    # Create all required directories
    all_dirs = [args.log_dir, args.ckpt_dir, args.data_dir, args.antigen_esm_dir, args.fold_emb_dir]
    for directory in all_dirs:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created or already exists: {directory}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)

    # Initialize model with CUDA check
    if args.cuda and torch.cuda.is_available():
        model = antibinder(antibody_hidden_dim=1024, antigen_hidden_dim=1024, latent_dim=args.latent_dim, res=False).cuda()
    else:
        model = antibinder(antibody_hidden_dim=1024, antigen_hidden_dim=1024, latent_dim=args.latent_dim, res=False)
    
    print(model)

    # Dataset path selection
    if args.data == 'train':
        train_path = os.path.join(args.data_dir, 'combined_training_data.csv')
        test_path = os.path.join(args.data_dir, 'test_data.csv')
    # Additional dataset options can be added here
    
    # Ensure train and test data files exist
    if not os.path.exists(train_path):
        print(f"Warning: Train data file not found at {train_path}")
        print("Please ensure the train dataset exists at the specified location.")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"Warning: Test data file not found at {test_path}")
        print("Please ensure the test dataset exists at the specified location.")
        sys.exit(1)
    else:
        print(f"Using train dataset: {train_path}")
        print(f"Using test dataset: {test_path}")

    # Try to initialize dataset with appropriate parameters
    try:
        train_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            data_path=train_path, 
            train=True, 
            test=False, 
            rate1=1,
        )
        test_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            data_path=test_path,
            train=False,
            test=True,
            rate1=1,
        )
        
    except TypeError as e:
        print(f"Error with dataset initialization: {e}")
        print("Trying alternative initialization...")
        # Use only the parameters that are actually defined in the class
        train_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            data_path=train_path, 
            train=True, 
            test=False, 
            rate1=1
        )
        test_dataset = antibody_antigen_dataset(
            antigen_config=antigen_config,
            antibody_config=antibody_config,
            data_path=test_path,
            train=False,
            test=True,
            rate1=1,
        )
    def custom_collate(batch):
        print(f"Collate function received batch with {len(batch)} samples")
        if len(batch) > 0:
            print(f"First sample has {len(batch[0])} elements")
            for i, element in enumerate(batch[0]):
                print(f"  Sample element {i}: type={type(element)}, shape={getattr(element, 'shape', 'no shape')}")
        
        try:
            # Handle the case where dataset returns 4 elements instead of 3
            if len(batch[0]) == 4:
                # Unpack 4 elements and ignore the 4th one (assuming it's not needed)
                antibody_sets, antigen_sets, labels, extra = zip(*batch)
                print(f"Unpacked 4 elements: antibody_sets={len(antibody_sets)}, antigen_sets={len(antigen_sets)}, labels={len(labels)}, extra={len(extra)}")
                print(f"Extra element type: {type(extra[0])}, value: {extra[0]}")
            elif len(batch[0]) == 3:
                # Original expected format
                antibody_sets, antigen_sets, labels = zip(*batch)
                print(f"Unpacked 3 elements: antibody_sets={len(antibody_sets)}, antigen_sets={len(antigen_sets)}, labels={len(labels)}")
            else:
                raise ValueError(f"Unexpected number of elements in batch: {len(batch[0])}")
            
            antibody_sets = [torch.stack(x) for x in zip(*antibody_sets)]
            antigen_sets = [torch.stack(x) for x in zip(*antigen_sets)]
            labels = torch.stack(labels)
            
            print(f"Final output: antibody_sets={len(antibody_sets)} tensors, antigen_sets={len(antigen_sets)} tensors, labels shape={labels.shape}")
            return antibody_sets, antigen_sets, labels
        except Exception as e:
            print(f"Error in custom_collate: {e}")
            import traceback
            traceback.print_exc()
            raise

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate)
    valid_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=custom_collate)
    
    # Debug: Check the first batch to understand data structure
    print("Debugging dataloader output...")
    try:
        first_batch = next(iter(train_dataloader))
        print(f"First batch has {len(first_batch)} elements")
        for i, item in enumerate(first_batch):
            if hasattr(item, '__len__') and not isinstance(item, torch.Tensor):
                print(f"Element {i}: type={type(item)}, length={len(item)}")
                if len(item) > 0:
                    print(f"  First sub-element type: {type(item[0])}")
                    if hasattr(item[0], 'shape'):
                        print(f"  First sub-element shape: {item[0].shape}")
            else:
                print(f"Element {i}: type={type(item)}, shape={getattr(item, 'shape', 'no shape')}")
    except Exception as e:
        print(f"Error checking first batch: {e}")
        import traceback
        traceback.print_exc()
    
    print("Starting training...")
    
    # Create logger with path using the directory we ensured exists
    log_file = os.path.join(
        args.log_dir, 
        f"{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv"
    )
 
    logger = CSVLogger_my(
        ['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall',
        'val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall'],
        log_file
    )   
    

    # Model loading section with error handling
    load = False
    if load:
        try:
            weight_path = ''  # Path should be specified when load=True
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, 
                                       map_location=None if (args.cuda and torch.cuda.is_available()) else torch.device('cpu'))
                
                # Handle both old and new checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')} with val_loss {checkpoint.get('val_loss', 'unknown')}")
                else:
                    # Old format - just state dict
                    model.load_state_dict(checkpoint)
                    print("Model loaded successfully (old format)")
            else:
                print(f"Warning: Model file not found at {weight_path}")
                load = False
        except Exception as e:
            print(f"Error loading model: {e}")
            load = False

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,  # Pass your test/validation dataloader here
        logger=logger,
        args=args,
        load=load
    )

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)