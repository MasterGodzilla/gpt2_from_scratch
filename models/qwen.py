import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel    

from typing import Union
from dataclasses import dataclass

@dataclass
class QwenConfig:
    block_size: int = 32768
    vocab_size: int = 151936
    n_layer: int = 24
    n_kv_head: int = 2
    n_q_head: int = 14
    n_embd: int = 896
    rope_base: int = 1000000
    tie_embeddings: bool = True # tie the token embedding and lm head

qwen_config_dict = {
    'qwen05b':    dict(n_layer=24, n_kv_head=2, n_q_head=14, n_embd=896, rope_base=1000000, block_size=32768, tie_embeddings=True),
    'qwen15b':    dict(n_layer=28, n_kv_head=2, n_q_head=12, n_embd=1024, block_size=32768, tie_embeddings=True),
    'qwen3b':     dict(n_layer=36, n_kv_head=2, n_q_head=16, n_embd=1536, block_size=32768, tie_embeddings=True),
    'qwen7b':     dict(n_layer=28, n_kv_head=4, n_q_head=28, n_embd=4096, block_size=128000, tie_embeddings=False),
    'qwen14b':    dict(n_layer=48, n_kv_head=8, n_q_head=40, n_embd=5120, block_size=128000, tie_embeddings=False),
    'qwen32b':    dict(n_layer=64, n_kv_head=8, n_q_head=40, n_embd=6144, block_size=128000, tie_embeddings=False),
    'qwen72b':    dict(n_layer=80, n_kv_head=8, n_q_head=64, n_embd=8192, block_size=128000, tie_embeddings=False),
}


class TransformerBlock(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias = True) # up
        self.gelu = nn.GELU(approximate = 'tanh') # activation func
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd, bias=True) # down
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module): 

    def __init__(self, config: QwenConfig): 
        super().__init__()
        self.config = config

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias = True) # Q, K, V, [768, 2304]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = True) # proj back to embedding

        # tril. 1, 1 for batch_size and head dim
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x): 
        # B: batch_size, T: block_size, C: n_embd
        # Note: T is the block_size of the current batch, which can be smaller than config.block_size
        B, T, C = x.size()
        H = self.config.n_head
        qkv = self.c_attn(x) # [batch_size, block_size, n_embd] --> [batch_size, block_size, n_embd*3]
        q, k, v = qkv.split(self.config.n_embd, dim = 2)
        k = k.view(B, T, H, C//H).transpose(1,2) # B, T, C --> B, T, H, C//H --> B, H, T, C//H
        q = q.view(B, T, H, C//H).transpose(1,2) # B, T, C --> B, T, H, C//H --> B, H, T, C//H
        v = v.view(B, T, H, C//H).transpose(1,2) # B, T, C --> B, T, H, C//H --> B, H, T, C//H

        # A = softmax((QK^T)/\sqrt{d})
        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, H, T, C//H) @ (B, H, C//H, T) = (B, H, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, H, T, T) @ (B, H, T, C//H) --> (B, H, T, C//H)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) #  (B, H, T, C//H) --> (B, T, H, C//H) --> (B, T, C)
        y = self.c_proj(y)
        return y


class Qwen(nn.Module):
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word embedding table
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding table
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]), # transformer layers
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scale the c_proj weights by sqrt(layer_number)
        for param_name, param in self.named_parameters():
            if param_name.endswith('.c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 * math.sqrt(self.config.n_layer))
            
        print("number of parameters: ", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'qwen05b', 'qwen15b', 'qwen3b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen72b'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = qwen_config_dict[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = QwenConfig(**config_args)
        model = Qwen(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type.startswith('cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
                      
    def forward(self, x, y=None): 
        B, T = x.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the GPT model
        token_embeddings = self.transformer.wte(x)
        position_embeddings = self.transformer.wpe(torch.arange(T, device=x.device))
        x = token_embeddings + position_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
