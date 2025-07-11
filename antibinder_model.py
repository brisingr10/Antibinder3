import torch
import torch.nn as nn
from cfg_ab import configuration

class AntiModelInitial(nn.Module):
    def __init__(self, initializer_range=0.02):
        super().__init__()
        self.initializer_range = initializer_range

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, res=False):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.res = res

    def forward(self, query_embed, kv_embed, key_padding_mask=None):
        attn_output, _ = self.cross_attention(query=query_embed, key=kv_embed, value=kv_embed, key_padding_mask=key_padding_mask)
        if self.res:
            attn_output = attn_output + query_embed
        return self.norm(attn_output)

class AntiEmbeddings(nn.Module):
    def __init__(self, config: configuration, is_light_chain=False):
        super().__init__()
        self.config = config
        self.is_light_chain = is_light_chain
        
        token_size = config.token_size
        hidden_size = config.hidden_size
        type_residue_size = config.type_residue_size_light if is_light_chain else config.type_residue_size

        self.residue_embedding = nn.Embedding(token_size, hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(type_residue_size + 1, hidden_size) # Add 1 for padding index 0
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, seq_tokens, region_indices):
        seq_embeddings = self.residue_embedding(seq_tokens)
        token_type_embeddings = self.token_type_embeddings(region_indices)
        embeddings = seq_embeddings + token_type_embeddings
        return self.dropout(self.layer_norm(embeddings))

class Pooler(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dense(x))

class Combine_Embedding(nn.Module):
    def __init__(self, config: configuration):
        super().__init__()
        self.config = config
        self.heavy_chain_emb = AntiEmbeddings(config, is_light_chain=False)
        self.light_chain_emb = AntiEmbeddings(config, is_light_chain=True)
        self.antigen_emb = AntiEmbeddings(config) # Antigen doesn't have region types in this setup

        self.antibody_structure_proj = nn.Linear(64, config.hidden_size)
        self.antigen_structure_proj = nn.Linear(1280, config.hidden_size)

    def _pad_or_truncate(self, tensor, target_length):
        """Pad or truncate tensor to target length along dimension 1"""
        current_length = tensor.shape[1]
        if current_length == target_length:
            return tensor
        elif current_length < target_length:
            # Pad with zeros
            pad_length = target_length - current_length
            padding = torch.zeros(tensor.shape[0], pad_length, tensor.shape[2], device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=1)
        else:
            # Truncate
            return tensor[:, :target_length, :]

    def forward(self, heavy_chain, light_chain, antigen):
        # Move data to appropriate device (GPU if available, CPU otherwise)
        device = next(self.parameters()).device
        for k, v in heavy_chain.items(): heavy_chain[k] = v.to(device)
        for k, v in light_chain.items(): light_chain[k] = v.to(device)
        for k, v in antigen.items(): antigen[k] = v.to(device)

        # Get sequence embeddings
        heavy_seq_emb = self.heavy_chain_emb(heavy_chain['tokens'], heavy_chain['regions'])
        light_seq_emb = self.light_chain_emb(light_chain['tokens'], light_chain['regions'])
        antigen_seq_emb = self.antigen_emb(antigen['tokens'], torch.zeros_like(antigen['tokens'])) # Zeroes for region

        # Project and add structure embeddings
        heavy_structure_emb = self.antibody_structure_proj(heavy_chain['structure'])
        light_structure_emb = self.antibody_structure_proj(light_chain['structure'])
        antigen_structure_emb = self.antigen_structure_proj(antigen['structure'])

        # Pad/truncate structure embeddings to match sequence lengths
        heavy_structure_emb = self._pad_or_truncate(heavy_structure_emb, self.config.max_position_embeddings)
        light_structure_emb = self._pad_or_truncate(light_structure_emb, self.config.max_position_embeddings_light)
        antigen_structure_emb = self._pad_or_truncate(antigen_structure_emb, self.config.max_position_embeddings)

        heavy_total_emb = heavy_seq_emb + heavy_structure_emb
        light_total_emb = light_seq_emb + light_structure_emb
        antigen_total_emb = antigen_seq_emb + antigen_structure_emb

        # Concatenate heavy and light chains
        combined_antibody_emb = torch.cat((heavy_total_emb, light_total_emb), dim=1)
        
        return combined_antibody_emb, antigen_total_emb

class BiCrossAttentionBlock(nn.Module):
    def __init__(self, config: configuration, latent_dim=64, res=False):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        self.ab_to_ag_attention = BidirectionalCrossAttention(config.hidden_size, num_heads=1, res=res)
        self.ag_to_ab_attention = BidirectionalCrossAttention(config.hidden_size, num_heads=1, res=res)

        combined_ab_len = config.max_position_embeddings + config.max_position_embeddings_light
        self.ab_pooler = Pooler(combined_ab_len * config.hidden_size, latent_dim * latent_dim)
        self.ag_pooler = Pooler(config.max_position_embeddings * config.hidden_size, latent_dim * latent_dim)

        self.flatten = nn.Flatten()
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, antibody_emb, antigen_emb):
        ab_attended = self.ab_to_ag_attention(antibody_emb, antigen_emb)
        ag_attended = self.ag_to_ab_attention(antigen_emb, antibody_emb)

        ab_pooled = self.ab_pooler(self.flatten(ab_attended)).view(-1, self.latent_dim, self.latent_dim)
        ag_pooled = self.ag_pooler(self.flatten(ag_attended)).view(-1, self.latent_dim, self.latent_dim)

        return self.flatten(ab_pooled), self.flatten(ag_pooled)

class AntiBinder(nn.Module):
    def __init__(self, config: configuration, latent_dim=32, res=False):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        self.embedding_combiner = Combine_Embedding(config)
        self.cross_attention_block = BiCrossAttentionBlock(config, latent_dim, res)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * latent_dim * 2, latent_dim * latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim * latent_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(AntiModelInitial()._init_weights)

    def forward(self, heavy_chain, light_chain, antigen):
        antibody_emb, antigen_emb = self.embedding_combiner(heavy_chain, light_chain, antigen)
        ab_pooled, ag_pooled = self.cross_attention_block(antibody_emb, antigen_emb)
        
        # Combine pooled representations
        alpha = self.cross_attention_block.alpha
        concatenated = torch.cat((ab_pooled, alpha * ag_pooled), dim=-1)
        
        return self.classifier(concatenated)

if __name__ == '__main__':
    # --- Config ---
    config = configuration()
    
    # --- Model ---
    model = AntiBinder(config, latent_dim=32, res=True)
    
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device)
        print(f"Model created successfully on {torch.cuda.device_count()} GPU(s).")
    else:
        model = model.to(device)
        print("Model created successfully on CPU.")

    # --- Dummy Data ---
    batch_size = 4
    heavy_chain_data = {
        'tokens': torch.randint(0, 21, (batch_size, config.max_position_embeddings)),
        'regions': torch.randint(0, 7, (batch_size, config.max_position_embeddings)),  # 0-6 for heavy chain
        'structure': torch.rand(batch_size, config.max_position_embeddings, 64)
    }
    light_chain_data = {
        'tokens': torch.randint(0, 21, (batch_size, config.max_position_embeddings_light)),
        'regions': torch.randint(1, 7, (batch_size, config.max_position_embeddings_light)),  # 1-6 for light chain (0 reserved for padding)
        'structure': torch.rand(batch_size, config.max_position_embeddings_light, 64)
    }
    antigen_data = {
        'tokens': torch.randint(0, 21, (batch_size, config.max_position_embeddings)),
        'structure': torch.rand(batch_size, config.max_position_embeddings, 1280)
    }

    # --- Forward Pass ---
    output = model(heavy_chain_data, light_chain_data, antigen_data)
    print("Forward pass successful, output shape:", output.shape)
    print(f"Model running on device: {device}")