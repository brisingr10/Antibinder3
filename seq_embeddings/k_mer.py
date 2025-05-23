import lmdb 
import pickle
from collections import defaultdict 
from itertools import product 
import torch 
import hashlib


def get_consistent_hash_key(input_string) :
    hash_object = hashlib.md5(input_string.encode('utf-8'))
    return hash_object.hexdigest()


def calculate_kmer_frequencies(sequence, k):
    kmer_counts = defaultdict(int)
    for i in range(len(sequence) - k + 1):
        kmer = sequence [i:i+k]
        kmer_counts[kmer] += 1
    return {kmer: count / (len(sequence) - k + 1) for kmer, count in kmer_counts.items()}


def generate_possible_kmers(k, alphabet='ACDEFGHIKLMNPQRSTVWY'):
    return [''. join(p) for p in product(alphabet, repeat=k)]


def kmer_frequencies_to_feature_vector(kmer_freqs, k, alphabet= 'ACDEFGHIKLMNPQRSTVWY') :
    possible_kmers = generate_possible_kmers(k, alphabet)
    return [kmer_freqs.get(kmer, 0) for kmer in possible_kmers]


def save_feature_vector_to_lmdb(feature_vector, lmdb_env, key):
    with lmdb_env.begin(write=True) as txn:
        txn.put(key.encode('utf-8'), pickle.dumps(feature_vector))
                    

def load_feature_vector_from_lmdb(lmdb_env, key) :
    with lmdb_env.begin(write=False) as txn:
        value = txn.get(key.encode('utf-8'))
        if value is not None:
            return pickle.loads(value)
        else:
            return None
        
def get_or_compute_feature_vector(sequence, k, lmdb_env, key):
    feature_vector = load_feature_vector_from_lmdb(lmdb_env, key)
    if feature_vector is None:
        kmer_freqs = calculate_kmer_frequencies(sequence, k)
        feature_vector = kmer_frequencies_to_feature_vector(kmer_freqs, k)
        save_feature_vector_to_lmdb(feature_vector, lmdb_env, key) 
    return feature_vector