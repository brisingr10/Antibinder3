from itertools import chain, product
import hashlib
import pickle
import numpy as np

amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(sequence, max_length=149):
    num_amino_acids = len(amino_acids)
    one_hot = np.zeros((max_length, num_amino_acids), dtype=int)
    
    fill_length = min(len(sequence), max_length)
    
    for i in range(fill_length):
        aa = sequence[i]
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
        else:
            raise ValueError(f"Invalid amino acid: {aa}")
    
    return one_hot


def save_onehot_vector_to_lmdb(feature_vector, lmdb_env, key):
    with lmdb_env.begin(write=True) as txn:
        txn.put(key.encode('utf-8'), pickle.dumps(feature_vector))


def load_onehot_vector_from_lmdb(lmdb_env, key) :
    with lmdb_env.begin(write=False) as txn:
        value = txn.get(key.encode('utf-8'))
        if value is not None:
            return pickle.loads(value)
        else:
            return None
        

def get_or_compute_onehot_vector(sequence, max_length, lmdb_env, key):
    feature_vector = load_onehot_vector_from_lmdb(lmdb_env, key)
    if feature_vector is None:
        feature_vector = one_hot_encode(sequence, max_length)
        save_onehot_vector_to_lmdb(feature_vector, lmdb_env, key) 
    return feature_vector