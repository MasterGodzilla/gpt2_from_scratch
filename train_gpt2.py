from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

config_dict = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-mini':    dict(n_layer=16, n_head=8, n_embd=512),  # 77M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
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

    def __init__(self, config: GPTConfig):
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

    def __init__(self, config: GPTConfig): 
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


class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
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
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = config_dict[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
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
        use_fused = fused_available and device_type == 'cuda'
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
    

from typing import Union
def generate(model: GPT, 
             x: Union[torch.Tensor, str, None] = None,
             tokenizer = None,
             max_new_tokens: int = 100, 
             temperature: float = 1.0, 
             top_k: int = None,
             device: str = 'cuda'):
    
    if isinstance(x, str):
        x = torch.tensor(tokenizer.encode(x)).unsqueeze(0)
    
    if isinstance(x, torch.Tensor): # so we catch error if str treatment fails
        pass
    else:
        raise ValueError(f"Invalid input type: {type(x)}")
    
    
    x = x.to(device) # [batch_size, seq_len]
    seq_len = x.size(1)
    if seq_len > model.config.block_size:
        raise ValueError(f"Cannot generate more tokens than the model's block size: {model.config.block_size}")
    max_new_tokens = max_new_tokens if max_new_tokens <= model.config.block_size - seq_len else model.config.block_size - seq_len
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        if temperature > 0.01:
            probs = F.softmax(logits / temperature, dim=-1)
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
            next_token = torch.multinomial(probs, num_samples=1)
        else: # greedy sampling
            next_token = torch.argmax(logits, dim=-1)
        x = torch.cat((x, next_token), dim=-1)
    return x



def test_model(model, tokenizer, device):
    model.eval()
    print('Testing model inference...')
    message = "Hello, how are you?"
    output = generate(model, message, tokenizer, max_new_tokens=100, temperature=0.5, device=device)
    print(">>>", tokenizer.decode(output[0].tolist()))
    model.train()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='GPT-2 Training and Testing')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--model_type', type=str, default='gpt2', choices=['gpt2', 'gpt2-mini', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Model type to use')
    parser.add_argument('--total_batch_size', type=int, default=2**19, help='Total batch size in tokens')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size in number of sequences')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--max_steps', type=int, default=30, help='Number of steps')
    parser.add_argument('--interval_size', type=int, default=5, help='Interval size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--learning_rate_schedule', type=str, default='cosine', choices=['cosine', 'trapezoid'], help='Learning rate schedule')
    return parser.parse_args()

class DataLoaderLite:
    def __init__(self, B, T, enc):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        self.tokens = enc.encode(text)
        print(f"Loaded {len(self.tokens)} tokens")

        self.current_idx = 0
        self.epoch = 0
    
    def next_batch(self, i=None):
        if i is None:
            i = self.current_idx
        seq_len = self.B * self.T
        start_idx = i
        end_idx = start_idx + seq_len + 1  # +1 for the target shift
        
        if end_idx >= len(self.tokens):
            self.current_idx = 0
            start_idx = 0
            end_idx = start_idx + seq_len + 1
            self.epoch += 1
            print(f"Completed epoch {self.epoch}")
            
        buf = torch.tensor(self.tokens[start_idx:end_idx])
        x = buf[:-1].view(self.B, self.T)  # Input sequence
        y = buf[1:].view(self.B, self.T)   # Target sequence
        self.current_idx = end_idx - 1  # -1 to rewind the target shift
        
        return x.to(args.device), y.to(args.device)

def get_lr(i, lr = 1e-3, schedule_type='cosine'):
    """
    Learning rate schedules:
    - "cosine": cosine learning rate schedule
    - "trapezoid": trapezoid learning rate schedule, where the learning rate is linearly increased from 0 to 1e-3 for the first 1% steps, keep at 1e-3 for the next 98% steps, and linearly decreased to 1e-4 for the last 1% steps
    """
    # consine learning rate schedule
    # 1) linear warmup for first 1% steps
    if i < args.max_steps // 100:
        return lr * i / (args.max_steps // 100)
    # 2) cosine learning rate decay
    if schedule_type == 'cosine':
        decay_ratio = (i - args.max_steps // 100) / (args.max_steps - args.max_steps // 100)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr * coeff + lr/10 * (1.0 - coeff)  # max_lr * coeff + min_lr * (1-coeff)
    elif schedule_type == 'trapezoid':
        # Linear warmup for first 1% steps
        warmup_steps = args.max_steps // 100
        if i < warmup_steps:
            return lr * i / warmup_steps
        # Constant lr for middle 98% steps 
        elif i < args.max_steps * 99 // 100:
            return lr
        # Linear decay for last 1% steps
        else:
            steps_left = args.max_steps - i
            return lr/10 + steps_left * (lr - lr/10) / (args.max_steps // 100)

if __name__ == "__main__":
    # terminal command: 
    # CUDA_VISIBLE_DEVICES=0 python train_gpt2.py --train
    # CUDA_VISIBLE_DEVICES=0 python train_gpt2.py --train --model_type gpt2-mini --batch_size 16 --learning_rate_schedule trapezoid
    
    args = parse_args()
    
    if not args.train and not args.test:
        print("Please specify either --train or --test")
        exit(1)
    
    if args.train:
        B, T = args.batch_size, args.seq_len
        torch.set_float32_matmul_precision('high')

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        data_loader = DataLoaderLite(B, T, enc)

        config_args = config_dict[args.model_type]
        config_args['vocab_size'] = 50304
        model = GPT(GPTConfig(**config_args))
        model.to(args.device)
        model.train()
        # model.compile()

        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type=args.device)

        assert args.total_batch_size % (B * T) == 0, "Total batch size must be divisible by the product of batch size and sequence length"
        grad_accum_steps = args.total_batch_size // (B * T)
        from time import time
        for i in range(args.max_steps):
            if i % args.interval_size == 0:
                test_model(model, enc, args.device)

            start_time = time()
            optimizer.zero_grad()
            loss_accum = 0.0
            for _ in range(grad_accum_steps):
                x, y = data_loader.next_batch()
                with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(i, args.learning_rate, args.learning_rate_schedule)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            torch.cuda.synchronize()
            end_time = time()
            throughput = (args.total_batch_size) / (end_time - start_time)
            print(f'iter {i}, loss: {loss_accum:.4f}, time: {end_time - start_time:.4f}s, norm: {norm:.4f}, lr: {lr:.2e}, throughput: {throughput:.1f} tokens/s')

    if args.test:
        device = args.device
        model = GPT.from_pretrained(args.model_type)
        model.to(device)
        model.eval()

        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")

        test_model(model, tokenizer, device)
       