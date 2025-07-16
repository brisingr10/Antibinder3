import hashlib
import esm
import os
import lmdb
import pickle
import torch
import torch.nn as nn
import pandas as pd
from cfg_ab import AminoAcid_Vocab, configuration
from igfold import IgFoldRunner

# --- Fix SSL certificate issue on macOS ---
import ssl
import certifi
try:
    # Try to use certifi certificates for SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

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

# --- Setup Paths ---
PROJECT_ROOT = os.path.dirname(__file__)
ANTIGEN_ESM_CACHE_DIR = os.path.join(PROJECT_ROOT, 'antigen_esm', 'train')
FOLD_EMB_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'fold_emb')
HEAVY_CHAIN_CACHE = os.path.join(FOLD_EMB_DIR, 'heavy_chain_structures')
LIGHT_CHAIN_CACHE = os.path.join(FOLD_EMB_DIR, 'light_chain_structures')

# --- Create Cache Directories ---
os.makedirs(ANTIGEN_ESM_CACHE_DIR, exist_ok=True)
os.makedirs(HEAVY_CHAIN_CACHE, exist_ok=True)
os.makedirs(LIGHT_CHAIN_CACHE, exist_ok=True)

class antibody_antigen_dataset(nn.Module):
    def __init__(self, antigen_config: configuration, antibody_config: configuration, data_path=None, train=True, test=False, data=None):
        super().__init__()
        self.antigen_config = antigen_config
        self.antibody_config = antibody_config
        self.test = test  # Store test flag for use in __getitem__

        # --- Load Data ---
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.read_csv(data_path)

        # --- Data Validation ---
        heavy_chain_cols = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
        light_chain_cols = ['L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4']
        df.dropna(subset=heavy_chain_cols + light_chain_cols, inplace=True)
        self.data = df

        # --- ESM and IgFold Models ---
        self.antigen_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        self.igfold = IgFoldRunner()

        # --- LMDB Caches ---
        self.heavy_chain_env = lmdb.open(HEAVY_CHAIN_CACHE, map_size=1024**3 * 50, lock=False)
        self.light_chain_env = lmdb.open(LIGHT_CHAIN_CACHE, map_size=1024**3 * 50, lock=False)

        if train:
            print(f"Dataset for train with {len(self.data)} samples.")
        if test:
            print(f"Dataset for test with {len(self.data)} samples.")

    # --- Padding Utilities ---
    def _pad_sequence(self, sequence, max_length):
        return torch.cat([sequence, torch.zeros(max_length - len(sequence))]).long() if len(sequence) < max_length else sequence[:max_length]

    def _pad_for_esm(self, sequence, max_length):
        return sequence + '<pad>' * (max_length - len(sequence))

    # --- Region Indexing ---
    def _create_region_indices(self, data_row, chain_type: str):
        config = self.antibody_config
        if chain_type == 'heavy':
            regions = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
            mapping = config.heavy_chain_region_type_indexing
            max_len = config.max_position_embeddings
        else: # light
            regions = ['L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4']
            mapping = config.light_chain_region_type_indexing
            max_len = config.max_position_embeddings_light

        indices = []
        for region in regions:
            # Use a default value from the mapping if a specific key like H-FR4 is not present
            region_id = mapping.get(region, 1) # Default to FR-like index
            indices.extend([region_id] * len(data_row[region]))
        
        return self._pad_sequence(torch.tensor(indices), max_len)

    # --- Structure Embedding ---
    def _get_structure_embedding(self, sequence, chain_type, lmdb_env):
        with lmdb_env.begin(write=True) as txn:
            structure_bytes = txn.get(sequence.encode())
            if structure_bytes is None:
                sequences = {"H" if chain_type == 'heavy' else "L": sequence}
                emb = self.igfold.embed(sequences=sequences)
                structure = emb.structure_embs.detach().cpu()
                txn.put(sequence.encode(), pickle.dumps(structure))
            else:
                structure = pickle.loads(structure_bytes)
        return structure

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        
        # Handle label - use dummy label for test mode when ANT_Binding doesn't exist
        if self.test and 'ANT_Binding' not in data_row:
            label = torch.tensor(0)  # Dummy label for test mode
        else:
            label = torch.tensor(data_row['ANT_Binding'])

        # --- Antigen Processing ---
        antigen_seq = str(data_row['Antigen Sequence'])
        antigen_cache_path = os.path.join(ANTIGEN_ESM_CACHE_DIR, hashlib.md5(antigen_seq.encode()).hexdigest() + '.pt')
        if not os.path.exists(antigen_cache_path):
            padded_antigen = self._pad_for_esm(antigen_seq, self.antigen_config.max_position_embeddings)
            batch_labels, batch_strs, antigen_tokens = self.batch_converter([('antigen', padded_antigen)])
            with torch.no_grad():
                results = self.antigen_model(antigen_tokens, repr_layers=[33], return_contacts=True)
            antigen_structure = results['representations'][33].squeeze(0)
            torch.save(antigen_structure, antigen_cache_path)
        else:
            antigen_structure = torch.load(antigen_cache_path)

        # --- Heavy Chain Processing ---
        vh_seq = "".join([str(data_row[col]) for col in ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']])
        vh_seq = vh_seq.upper().replace('O', 'X')  # Handle non-standard amino acids
        vh_seq = ''.join([aa if aa in AminoAcid_Vocab else 'X' for aa in vh_seq])  # Replace unknown AA with X
        vh_structure = self._get_structure_embedding(vh_seq, 'heavy', self.heavy_chain_env).squeeze(0)
        vh_token_ids = self._pad_sequence(torch.tensor([AminoAcid_Vocab[aa] for aa in vh_seq]), self.antibody_config.max_position_embeddings)
        vh_region_indices = self._create_region_indices(data_row, 'heavy')

        # --- Light Chain Processing ---
        vl_seq = "".join([str(data_row[col]) for col in ['L-FR1', 'L-CDR1', 'L-FR2', 'L-CDR2', 'L-FR3', 'L-CDR3', 'L-FR4']])
        vl_seq = vl_seq.upper().replace('O', 'X')  # Handle non-standard amino acids
        vl_seq = ''.join([aa if aa in AminoAcid_Vocab else 'X' for aa in vl_seq])  # Replace unknown AA with X
        vl_structure = self._get_structure_embedding(vl_seq, 'light', self.light_chain_env).squeeze(0)
        vl_token_ids = self._pad_sequence(torch.tensor([AminoAcid_Vocab[aa] for aa in vl_seq]), self.antibody_config.max_position_embeddings_light)
        vl_region_indices = self._create_region_indices(data_row, 'light')

        # --- Antigen Sequence to Tokens ---
        antigen_seq = str(data_row['Antigen Sequence']).upper().replace('O', 'X')  # Handle non-standard amino acids
        antigen_seq = ''.join([aa if aa in AminoAcid_Vocab else 'X' for aa in antigen_seq])  # Replace unknown AA with X
        antigen_token_ids = self._pad_sequence(torch.tensor([AminoAcid_Vocab[aa] for aa in antigen_seq]), self.antigen_config.max_position_embeddings)

        return {
            "heavy_chain": {"tokens": vh_token_ids, "regions": vh_region_indices, "structure": vh_structure},
            "light_chain": {"tokens": vl_token_ids, "regions": vl_region_indices, "structure": vl_structure},
            "antigen": {"tokens": antigen_token_ids, "structure": antigen_structure},
            "label": label,
            "index": index
        }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # Example usage
    # Set CUDA devices if available (can be overridden by environment variable)
    if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    
    antigen_config = configuration()
    antibody_config = configuration()
    data_path = os.path.join(PROJECT_ROOT, 'datasets', 'training_data.csv')
    
    dataset = antibody_antigen_dataset(
        antigen_config=antigen_config, 
        antibody_config=antibody_config, 
        data_path=data_path, 
        train=True
    )
    
    # Fetch a sample
    sample = dataset[0]
    print("Sample fetched successfully!")
    import pdb
    pdb.set_trace()
