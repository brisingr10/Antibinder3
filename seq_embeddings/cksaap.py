from itertools import chain, product
import hashlib
import pickle


AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DP = list(product(AA, AA))
DP_list = []
for i in DP:
    DP_list.append(str(i[0]) + str(i[1]))


def returnCKSAAPcode(query_seq, k):
    code_final = []
    for turns in range(k + 1):
        DP_dic = {}
        code = []
        code_order = []
        for i in DP_list:
            DP_dic[i] = 0
        for i in range(len(query_seq) - turns - 1):
            tmp_dp_1 = query_seq[i]
            tmp_dp_2 = query_seq[i + turns + 1]
            tmp_dp = tmp_dp_1 + tmp_dp_2
            if tmp_dp in DP_dic.keys():
                DP_dic[tmp_dp] += 1
            else:
                DP_dic[tmp_dp] = 1
        for i, j in DP_dic.items():
            code.append(j / (len(query_seq) - turns - 1))
        AAindex_list = DP_list[:]
        for i in AAindex_list:
            code_order.append(code[DP_list.index(i)])
        # code_final+=code_order
        code_final += code
    return code_final


def get_consistent_hash_key(input_string) :
    hash_object = hashlib.md5(input_string.encode('utf-8'))
    return hash_object.hexdigest()


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
        feature_vector = returnCKSAAPcode(sequence, k)
        save_feature_vector_to_lmdb(feature_vector, lmdb_env, key) 
    return feature_vector