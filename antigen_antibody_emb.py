import esm
import os
import lmdb
import pickle
import torch
import torch.nn as nn
import torch.nn. functional as F
import pandas as pd
import sys
from cfg_ab import AminoAcid_Vocab
from cfg_ab import configuration
import pdb
from math import ceil
from igfold import IgFoldRunner

# Get the directory where antigen_antibody_emb.py is located (project root)
PROJECT_ROOT = os.path.dirname(__file__) 
ANTIGEN_ESM_CACHE_DIR = os.path.join(PROJECT_ROOT, 'antigen_esm', 'train')
FOLD_EMB_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'fold_emb')

# Create cache directories if they don't exist
os.makedirs(ANTIGEN_ESM_CACHE_DIR, exist_ok=True)
# Note: LMDB directory needs to exist *before* opening. 
# Ensure 'fold_emb_for_train' exists within FOLD_EMB_DIR if using LMDB caching.
os.makedirs(os.path.join(FOLD_EMB_DIR, 'fold_emb_for_train'), exist_ok=True) # Uncomment if needed

class antibody_antigen_dataset(nn.Module):
    def __init__(self,
                antigen_config: configuration, 
                antibody_config: configuration, 
                data_path=None,
                train = True,
                test = False, 
                rate1 = 0.8,
                data = None) -> None:
        super().__init__()
        self.antigen_config = antigen_config
        self.antibody_config = antibody_config
        print (data_path)
        if isinstance(data,pd.DataFrame):
            df = data
            df.dropna()
        else:
            df = pd.read_csv(data_path)## samples of data, attention to your file type
            # df = df.dropna()
            df = df.dropna(subset=['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
   
        self.antigen_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()

        if train==True and test==False: # train
            print("This part of dataset is for train.")
            self.data = df.iloc[:int(df.shape[0]*(rate1))]
            print("len train data", len(self.data)) 

        elif train==False and test == True: # val
            print("This part of dataset is for test.")
            self.data = df.iloc[int(df.shape[0]*(rate1)):]
            print("len test data",len(self.data))

        self.env = None
        self.structure_embedding = None
        self.igfold = IgFoldRunner()


    def universal_padding(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return torch.cat([sequence,torch.zeros(max_length-len(sequence))]).long()
    

    def func_padding_for_esm(self, sequence, max_length):
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence+'<pad>'*(max_length-len(sequence)-2)
        

    def region_indexing(self,index):
        data = self.data.iloc[index]
        HF1 = [1 for _ in range(len(data['H-FR1']))]
        HCDR1 = [3 for _ in range(len (data['H-CDR1']))]
        HF2 = [1 for _ in range(len(data['H-FR2']))]
        HCDR2 = [4 for _ in range(len(data['H-CDR2']))]
        HF3 = [1 for _ in range(len(data['H-FR3']))]
        HCDR3 = [5 for _ in range(len(data['H-CDR3']))]
        HF4 = [1 for _ in range(len(data['H-FR4']))]
        vh = torch.tensor(list(HF1+HCDR1+HF2+HCDR2+HF3+HCDR3+HF4))
        vh = self.universal_padding(vh,self.antibody_config.max_position_embeddings)

        return vh
    

    def __getitem__(self, index):
        data = self.data.iloc[index]
        label = torch.tensor(data['ANT_Binding'])
        antigen_cache_filename = str(self.data.iloc[index]['Antigen Sequence']) + '.pt'
        antigen_cache_path = os.path.join(ANTIGEN_ESM_CACHE_DIR, antigen_cache_filename) # Use defined cache dir

        if not os.path.exists(antigen_cache_path): # Check using the constructed path
            antigen = self.func_padding_for_esm(self.data['Antigen Sequence'].iloc[index], self.antigen_config.max_position_embeddings)
            antigen = [('antigen', antigen)]
            batch_labels, batch_strs, antigen = self.batch_converter(antigen)
            with torch.no_grad():
                self.antigen_model = self.antigen_model.eval()
                antigen = self.antigen_model(antigen.squeeze(1), repr_layers=[33], return_contacts=True)
                antigen = antigen['representations'][33].squeeze(0)
            torch.save(antigen, antigen_cache_path) # Save using the constructed path
        
        antigen_structure = torch.load(antigen_cache_path) # Load using the constructed path
        # ...existing code...
        emb_seq = data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'] + data['H-CDR3']+data['H-FR4']
        #if not emb_seq in self.structure_embedding.keys():
        fold_emb_db_path = os.path.join(FOLD_EMB_DIR, 'fold_emb_for_train') # Use defined fold emb dir
        self.env = lmdb.open(fold_emb_db_path, map_size=1024*1024*1024*50, lock=False) # Use constructed path
        self.structure_embedding = self.env.begin(write=True)
        # ...existing code...
        if self.structure_embedding.get(emb_seq.encode()) == None:
            sequences = {
                "H": emb_seq
                }
            emb = self.igfold.embed(
                sequences=sequences,
                )
            structure = emb.structure_embs.detach().cpu()
            #self.structure_embedding[emb_seq] = structure
            self.structure_embedding.put(key=emb_seq.encode(), value=pickle.dumps(structure)) 
            self.structure_embedding.commit()
        else:
            # print("Structure is not none.")
            structure = pickle.loads(self.structure_embedding.get(emb_seq.encode()))
            # print("stru"，structure)
            # print("stru.shape"，structure.shape)
        self.env.close()

        structure_m1 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'])
        structure_m2 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'])
        structure_m3 = len(data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3']+ data['H-CDR3'] + data['H-FR4'])
        # print("zero_for_padding", zero_for_padding)
        # print("zero_for_padding.shape"，zero_for_padding.shape)
        part1_indices = torch.arange(structure_m1)
        part2_indices = torch.arange(structure_m1, structure_m2)
        part3_indices = torch.arange(structure_m2, structure_m3)

        part1 = structure.index_select(1, part1_indices)
        part2_full = structure.index_select(1, part2_indices)
        part2_mean = part2_full.mean(dim=1)
        part2_reshaped = part2_mean.view(part2_mean.size(0), 1, part2_mean.size(1))
        part3 = structure.index_select(1, part3_indices)

        structure = torch.cat((
            part1.detach().clone(),
            part2_reshaped.detach().clone(),
            part3.detach().clone()
        ), dim=1)
        # print("stru.shape", structure.shape)
        zero_for_padding = torch.zeros(1,self.antibody_config.max_position_embeddings-structure.shape[1], structure.shape[-1]).float()
        structure = torch.cat((structure, zero_for_padding) ,dim=1).squeeze(0)


        antibody = data['vh']
        # print("antibody："，antibody)
        antibody = torch.tensor([AminoAcid_Vocab[aa] for aa in antibody])
        # print("antibody："，antibody)
        antibody = self.universal_padding(sequence=antibody, max_length=self.antibody_config.max_position_embeddings)
        # print("antibody: ", antibody)
        # print("antibody_shape:", antibody. shape)

        at_type = self.region_indexing(index)
        # print(type)
        # print(type.shape)
        antibody_structure = structure

        antigen = data['Antigen Sequence']
        antigen = torch.tensor([AminoAcid_Vocab[aa] for aa in antigen])
        antigen = self.universal_padding(seq=antigen, maxlen=self.antigen_config.max_position_embeddings)

        antigen_structure = antigen_structure[:1024, :]
        # print(antigen_structure.shape)
        return [antibody,at_type,antibody_structure],[antigen,antigen_structure],label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    # ...existing code...
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    # Construct the data path relative to the project root
    data_path = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data.csv') 
    dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config, data_path=data_path, train=True, test=False, rate1=0.0001)
    # ...existing code...
    # pdb. set_trace()
    x1 = dataset[0]
 
    pdb.set_trace()

