import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = 'edu_fineweb10B'
dataset_name = 'HuggingFaceFW/fineweb-edu'
remote_name = 'sample-10BT'
cache_dir = '/workspace/cache' # cache dir for huggingface datasets
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fineweb_dataset = load_dataset(dataset_name, name=remote_name, cache_dir=cache_dir, num_proc=mp.cpu_count())

print (f"Loaded {len(fineweb_dataset['train'])} documents")

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot] # add eot token at the beginning
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < enc.n_vocab).all()
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

nprocs = max(1, os.cpu_count() // 2)
print (f"Using {nprocs} processes for tokenization")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fineweb_dataset['train'], chunksize=16):
        # if enough space in the current shard for the new tokens
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f"Sharding {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # if not enough space, write the current shard and start a new one
            split = 'val' if shard_index == 0 else 'train' # use the first shard for validation
            shard_path = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:] = tokens[:remainder]
            np.save(shard_path, all_tokens_np) # write_datafile()
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            shard_index += 1
            progress_bar = None
            # populate the next shard
            all_tokens_np[token_count:] = tokens[:remainder]
            token_count = len(tokens) - remainder

    # write the last shard
    split = 'val' if shard_index == 0 else 'train' # use the first shard for validation
    shard_path = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
    np.save(shard_path, all_tokens_np[:token_count])

    print(f"Total tokens: {token_count}")
    print(f"Total shards: {len(all_tokens_np) // shard_size}")


