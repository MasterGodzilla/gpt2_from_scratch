import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = 'edu_fineweb10B'
dataset_name = 'HuggingFaceFW/fineweb-edu'
remote_name = 'sample-10BT'
cache_dir = '/data2/cache' # cache dir for huggingface datasets
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fineweb_dataset = load_dataset(dataset_name, name=remote_name, cache_dir=cache_dir, num_proc=mp.cpu_count())

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    

