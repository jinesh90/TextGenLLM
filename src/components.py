import torch
import torch.nn as nn
from torch.utils.data import Dataset


# define GPT Data class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# define multihead attention class
class MultiHeadAttention(nn.Module):
  def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = nn.Dropout(dropout)
    self.context_length = context_length
    self.output = nn.Linear(d_out,d_out)

  def forward(self,x):
    no_of_batchs, total_words, d_in = x.shape

    # tensor shape no_of_batchs, total_words aka context_length , d_out
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)


    # this needed because to map independent head need dimention like no_of_batchs,self.num_heads,total_words,self.head_dim
    keys = keys.contiguous().view(no_of_batchs, total_words, self.num_heads, self.head_dim).transpose(1,2)
    queries = queries.contiguous().view(no_of_batchs, total_words, self.num_heads, self.head_dim).transpose(1,2)
    values = values.contiguous().view(no_of_batchs, total_words, self.num_heads, self.head_dim).transpose(1,2)

    atten_score = queries @ keys.transpose(-2,-1)

    # # this is crucial step for forward network.
    masked = torch.triu(torch.ones(total_words, total_words), diagonal=1).bool().to(x.device)
    atten_score.masked_fill(masked, float('-inf'))



    atten_weights = torch.softmax(atten_score / keys.shape[-1]**0.5, dim=-1)

    attn_weights = self.dropout(atten_weights)

    context_vec = (atten_weights @ values).transpose(1,2).contiguous().view(no_of_batchs, total_words, self.d_out)

    output = self.output(context_vec)

    return output

# building block helper class for transformers
class LayerNorm(nn.Module):
   def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

   def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    out = (x - mean) / torch.sqrt(var + self.eps)
    out = self.scale * out + self.shift
    return out

# GELU function as non linear activation.
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["layer_density"] * cfg["emb_dim"]),
            GELU(),
            nn.Linear(cfg["layer_density"] * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# transformer class
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# main GPT class
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits