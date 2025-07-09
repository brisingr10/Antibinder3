AminoAcid_Vocab = {
    "A":1,
    "R":2,
    "N":3, 
    "D":4,
    "C":5,
    "Q":6,
    "E":7,
    "G":8,
    "H":9,
    "I":10,
    "L":11,
    "K":12,
    "M":13,
    "F":14,
    "P":15,
    "S":16,
    "T":17,
    "W":18,
    "Y":19,
    "V":20,

    "X":21,  # <END>

}

class configuration():
    def __init__(self,
                    hidden_size: int = 768,
                    max_position_embeddings: int = 263, # For heavy chain
                    max_position_embeddings_light: int = 250, # For light chain
                    type_residue_size: int = 6, # For heavy chain (0-5)
                    type_residue_size_light: int = 5, # For light chain (6-10)
                    layer_norm_eps: float = 1e-12,
                    hidden_dropout_prob = 0.1,
                    use_bias = True,
                    initializer_range=0.02,
                    num_hidden_layers = 4,
                    type_embedding=False,
                    ) -> None:
        
        self.AminoAcid_Vocab = AminoAcid_Vocab
        self.token_size = len(self.AminoAcid_Vocab)
        self.residue_size = 21
        self.hidden_size = hidden_size
        self.pad_token_id = 0
        
        # Heavy Chain Config
        self.max_position_embeddings = max_position_embeddings
        self.type_residue_size = type_residue_size # 0 for padding, 1-5 for H-FR1, H-CDR1, H-FR2, H-CDR2, H-FR3

        # Light Chain Config
        self.max_position_embeddings_light = max_position_embeddings_light
        self.type_residue_size_light = type_residue_size_light # 6-10 for L-FR1, L-CDR1, L-FR2, L-CDR2, L-FR3

        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use__bias = use_bias
        self.num_hidden_layers = num_hidden_layers
        self.initializer_range = initializer_range
        self.type_embedding = type_embedding

        # Region type indexing
        self.heavy_chain_region_type_indexing = {
            'H-FR1': 1, 'H-CDR1': 2, 'H-FR2': 3, 'H-CDR2': 4, 'H-FR3': 5
        }
        self.vl_region_type_indexing = {
            'L-FR1': 6, 'L-CDR1': 7, 'L-FR2': 8, 'L-CDR2': 9, 'L-FR3': 10
        }