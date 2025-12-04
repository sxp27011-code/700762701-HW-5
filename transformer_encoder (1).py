import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention + residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward + residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


# ------------ Test Code ---------------
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 128

    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    model = SimpleTransformerEncoder(d_model=128, num_heads=8)

    out = model(x)

    print("Output shape:", out.shape)
