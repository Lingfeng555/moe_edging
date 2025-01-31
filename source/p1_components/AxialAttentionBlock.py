import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Attention block that processes a (batch_size, num_features, 12, 12) input
    and returns (batch_size, num_features, attention_value).
    """
    def __init__(self, num_features, attention_value=1, num_heads=4):
        super(AttentionBlock, self).__init__()
        
        self.num_features = num_features
        self.attention_value = attention_value
        self.num_heads = num_heads
        self.scale = (12 * 12) ** -0.5  # Scaling factor

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(12 * 12, 3 * (12 * 12), bias=False)
        self.out_proj = nn.Linear(12 * 12, attention_value)  # Output projection

    def forward(self, x):
        batch_size, num_features, h, w = x.shape  
        assert h == 12 and w == 12, "Expected input shape [batch_size, num_features, 12, 12]"
        
        x = x.view(batch_size, num_features, -1)  # Flatten spatial dimensions (12x12 → 144)
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)  # (batch, num_features, 144) → 3 tensors
        q, k, v = qkv

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # Scaled dot-product attention
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_out = torch.matmul(attn_probs, v)

        # Project attention output to the desired shape
        attn_out = self.out_proj(attn_out)

        return attn_out

# Example Usage
if __name__ == "__main__":
    batch_size, num_features = 32, 32
    x = torch.randn(batch_size, num_features, 12, 12)

    attention_block = AttentionBlock(num_features, attention_value=1)
    output = attention_block(x)
    print(output.shape) 