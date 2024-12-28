import os
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
from models.gpt2 import GPT, GPTConfig, gpt2_config_dict

from typing import Union
def generate(model: GPT, 
             x: Union[torch.Tensor, str, None] = None,
             tokenizer = None,
             max_new_tokens: int = 100, 
             temperature: float = 1.0, 
             top_k: int = None):
    
    if isinstance(x, str):
        x = torch.tensor(tokenizer.encode(x), device=next(model.parameters()).device).unsqueeze(0)
    
    if isinstance(x, torch.Tensor):
        x = x.to(next(model.parameters()).device)
    else:
        raise ValueError(f"Invalid input type: {type(x)}")
    
    
    # x = x.to(model.device) # [batch_size, seq_len]
    seq_len = x.size(1)
    if seq_len > model.config.block_size:
        raise ValueError(f"Cannot generate more tokens than the model's block size: {model.config.block_size}")
    max_new_tokens = max_new_tokens if max_new_tokens <= model.config.block_size - seq_len else model.config.block_size - seq_len
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[:, -1, :]
        # Mask out tokens beyond 50257 (the actual vocabulary size)
        logits[:, 50257:] = float('-inf')
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



def test_model(model, tokenizer):
    model.eval()
    print('Testing model inference...')
    message = "Hello, how are you?"
    output = generate(model, message, tokenizer, max_new_tokens=100, temperature=0.5)
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


import tiktoken
enc = tiktoken.get_encoding("gpt2")
import numpy as np
class DataLoaderLite:
    def __init__(self, B, T, local_rank = 0, world_size = 1, split = 'train'):
        self.B = B
        self.T = T
        self.local_rank = local_rank
        self.world_size = world_size
        self.split = split
        assert split in ['train', 'val']
        
        data_path = 'edu_fineweb10B'
        self.shards_paths = os.listdir(data_path)
        self.shards_paths = [s for s in self.shards_paths if s.startswith(f"edufineweb_{split}")]
        self.shards_paths.sort()
        self.shards_paths = [os.path.join(data_path, s) for s in self.shards_paths]
        assert len(self.shards_paths) > 0, f"No shards found for split {split}"

        if is_master_process:
            print(f"Loading {len(self.shards_paths)} shards for split {split}")

        self.current_shard_idx = 0
        self.tokens = torch.tensor(np.load(self.shards_paths[self.current_shard_idx]), dtype=torch.long)
        self.current_idx = local_rank * B * T # each process takes the ith chunk of a page of data
    
    def next_batch(self):
        seq_len = self.B * self.T 

        end_idx = self.current_idx + seq_len + 1  # +1 for the target shift
        
        if end_idx >= len(self.tokens):
            self.current_idx = self.local_rank * self.B * self.T
            end_idx = self.current_idx + seq_len + 1
            self.current_shard_idx += 1
            self.tokens = torch.tensor(np.load(self.shards_paths[self.current_shard_idx]), dtype=torch.long)
            if is_master_process:
                print(f"Completed shard {self.current_shard_idx} with {len(self.tokens)} tokens")
            
        buf = torch.tensor(self.tokens[self.current_idx:end_idx])
        x = buf[:-1].view(self.B, self.T)  # Input sequence
        y = buf[1:].view(self.B, self.T)   # Target sequence

        self.current_idx = self.current_idx + self.B * self.T * self.world_size
        return x.to(args.device), y.to(args.device)

def get_lr(i, lr = 1e-3, schedule_type='cosine'):
    """
    Learning rate schedules that handle small max_steps values:
    - "cosine": cosine learning rate schedule
    - "trapezoid": trapezoid learning rate schedule
    """
    # Calculate warmup steps (minimum of 1)
    warmup_steps = max(1, args.max_steps // 100)
    
    # Linear warmup
    if i < warmup_steps:
        return lr * i / warmup_steps
    
    if schedule_type == 'cosine':
        # Cosine decay from warmup to end
        decay_steps = max(1, args.max_steps - warmup_steps)
        decay_ratio = (i - warmup_steps) / decay_steps
        decay_ratio = min(1.0, max(0.0, decay_ratio))  # Clamp between 0 and 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr * coeff + lr/10 * (1.0 - coeff)  # max_lr * coeff + min_lr * (1-coeff)
    
    elif schedule_type == 'trapezoid':
        # Calculate phase boundaries
        cooldown_start = max(warmup_steps + 1, args.max_steps * 99 // 100)
        
        # Constant lr for middle phase
        if i < cooldown_start:
            return lr
            
        # Linear decay for final phase
        cooldown_steps = max(1, args.max_steps - cooldown_start)
        steps_left = args.max_steps - i
        return lr/10 + steps_left * (lr - lr/10) / cooldown_steps

if __name__ == "__main__":
    # terminal command: 
    # CUDA_VISIBLE_DEVICES=0 python train_gpt2.py --train
    # CUDA_VISIBLE_DEVICES=0 python train_gpt2.py --train --model_type gpt2-mini --batch_size 16 --learning_rate_schedule trapezoid
    
    args = parse_args()
    
    if not args.train and not args.test:
        print("Please specify either --train or --test")
        exit(1)
    
    

    if args.train:

        # ddp
        from torch.distributed import init_process_group, destroy_process_group
        import os

        # terminal command: 
        # CUDA_VISIBLE_DEVICES=0,5,8,9 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 torchrun --standalone --nproc_per_node=4 train_gpt2.py --train --model_type gpt2-mini --batch_size 16 --learning_rate_schedule trapezoid
        # torchrun --standalone --nproc_per_node=2 train_gpt2.py --train --model_type gpt2-mini --batch_size 16 --learning_rate_schedule trapezoid --total_batch_size 2**17 --max_steps 1000 --interval_size 50

        ddp = int(os.environ.get('RANK', -1)) != -1 # check if ddp is enabled
        if ddp:
            print("DDP enabled")

        if ddp:
            assert torch.cuda.is_available(), "We need cuda available for ddp!"
            assert 'cuda' in args.device, "Device types other than cuda is not supported by ddp"
            init_process_group(backend='nccl')
            ddp_rank = int(os.environ['RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(ddp_local_rank)
            print(f"Using device: {device}, rank: {ddp_rank}, world size: {ddp_world_size}, local rank: {ddp_local_rank}")
            is_master_process = ddp_rank == 0 
            
        else: 
            ddp_rank, ddp_local_rank, ddp_world_size, is_master_process = 0, 0, 1, True
            device = args.device
        
        torch.manual_seed(1337)
        if torch.cuda.is_available() and 'cuda' in args.device:
            torch.cuda.manual_seed(1337)    

        B, T = args.batch_size, args.seq_len
        assert args.total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by the product of batch size, sequence length, and world size"
        grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
        if is_master_process:
            print(f"Using grad accum steps: {grad_accum_steps}")
        
        torch.set_float32_matmul_precision('high')
        # enable tf32 for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        data_loader = DataLoaderLite(B, T, local_rank=ddp_local_rank, world_size=ddp_world_size, split='train')

        config_args = gpt2_config_dict[args.model_type]
        config_args['vocab_size'] = 50304
        model = GPT(GPTConfig(**config_args))
        model.to(device)
        model.train()
        model.compile()
        from torch.nn.parallel import DistributedDataParallel as DDP
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model with DDP

        raw_model = model.module if ddp else model
        optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type=device)

        
        from time import time
        assert args.max_steps < 1e10 / args.total_batch_size, f'Max steps must be less than 1e10 (tokens) / {args.total_batch_size} (batch size) = {int(1e10 / args.total_batch_size)} steps'
        for i in range(args.max_steps):
            if i % args.interval_size == 0 and is_master_process:
                test_model(raw_model, enc)

            start_time = time()
            optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(grad_accum_steps):
                x, y = data_loader.next_batch()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # sync gradients at the last micro step
                loss.backward()
            if ddp:
                torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(i, args.learning_rate, args.learning_rate_schedule)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            torch.cuda.synchronize()
            end_time = time()
            throughput = (args.total_batch_size) / (end_time - start_time)
            if is_master_process:
                print(f'iter {i}, loss: {loss_accum:.4f}, time: {end_time - start_time:.4f}s, norm: {norm:.4f}, lr: {lr:.2e}, throughput: {throughput:.1f} tokens/s')
        
        # save the model
        if is_master_process:
            print("Saving model...")
            torch.save(model.state_dict(), f"checkpoints/model_{args.model_type}.pth")

        if ddp:
            destroy_process_group()

    if args.test:
        model = GPT.from_pretrained(args.model_type)
        model.to(args.device)
        model.eval()

        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")

        test_model(model, tokenizer)
       
