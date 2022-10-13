import numpy as np
import torch
from einops import rearrange
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-base')

decode = np.memmap('decoder_output', mode='r', shape=(100000, 1))
retrieve = np.memmap('retrieved_tokens', mode='r', shape=(100000, 3))

print(decode[0])
print(retrieve[0])
print(tokenizer.decode(decode[0]))
print(tokenizer.decode(retrieve[0]))