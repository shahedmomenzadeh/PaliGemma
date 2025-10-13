from typing import Optional, Tuple
import torch
from torch import nn

class SigLipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = 0,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embeddings(pixel_values) # [Batch_size, Embed_dim, Num_Patches_H, Num_Patches_W]
        embeddings = patch_embeds.flatten(2) # [Batch Size, Embed_dim, Num_Patches]

        # We should transpose to match the standard transformer input: [Batch, Sequence Length, Embedding Dim]

        embeddings = embeddings.transpose(1, 2) # [Batch Size, Num_Patches, Embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings
    

class SiglipAttention(nn.Module):
    """ Multi Head Attention for SigLip """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_head = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_head
        self.scale = self.head_dim ** -0.5 # Equivalent to 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        # Create the attention layers
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        # [Batch_size, Num_patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [Batch_size, Num_patches, Embed_Dim] -> [Batch_size, Num_patches, Num_heads, Head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        # [Batch Size, Num Heads, Num Patches, Num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_head, seq_len, seq_len):

            raise ValueError(f"Unexpected attention weights shape: {attn_weights.size()}")
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_head, seq_len, self.head_dim):

            raise ValueError(f"Unexpected attention output shape: {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

 
class SiglipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config) # We will implement this later
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        # residual: [Batch Size, Num_Patches, Embed_dim]
        residual = hidden_states

        # [Batch Size, Num_Patches, Embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # [Batch Size, Num_Patches, Embed_dim] -> [Batch Size, Num_Patches, Embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)

        hidden_states = residual + hidden_states

        # Store hidden states for second residual connection
        residual = hidden_states # [Batch Size, Num_Patches, Embed_dim]
        hidden_states = self.layer_norm2(hidden_states)

        # [Batch Size, Num_Patches, Embed_dim] -> [Batch Size, Num_Patches, Embed_dim]
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states # Second residual connection

        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embedding = SigLipVisionEmbeddings(config) # We will implement this later -> Done
        self.encoder = SiglipEncoder(config) # We will implement this later
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_dim]
        hidden_states = self.embedding(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state



class SiglipVisionModel(nn.Module):

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)
    

