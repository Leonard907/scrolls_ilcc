# query 1000, memory: 32000, 320K, 3200K, topk: 32, 64, 128 random: randn, uniform
import faiss 
import numpy as np 
import time as time
D = 64
index = faiss.IndexFlatL2(D)
memory = [32000, 320000, 3200000]
topk = [32, 64, 128]
for mem in memory:
    for k in topk:
        print("memory: %d, topk: %d" % (mem, k))
        xq = np.random.randn(1000, D).astype('float32')
        xb = np.random.randn(mem, D).astype('float32')
        index.add(xb)
        start = time.time()
        result = index.search(xq, k)
        print('time: {}'.format(time.time() - start))
        index.reset()

index = faiss.IndexFlatIP(D)
memory = [32000, 320000, 3200000]
topk = [32, 64, 128]
for mem in memory:
    for k in topk:
        print("memory: %d, topk: %d" % (mem, k))
        xq = np.random.randn(1000, D).astype('float32')
        xb = np.random.randn(mem, D).astype('float32')
        index.add(xb)
        start = time.time()
        result = index.search(xq, k)
        print('time: {}'.format(time.time() - start))
        index.reset()