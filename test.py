# query 1000, memory: 32000, 320K, 3200K, topk: 32, 64, 128 random: randn, uniform
import faiss 
import numpy as np 
import time as time
D = 128
index = faiss.IndexFlatL2(D)
initialization = ['normal', 'uniform']
memory = [32000, 320000, 3200000]
topk = [32, 64, 128]
for init in initialization:
    for mem in memory:
        for k in topk:
            print('initialization: {}, memory: {}, topk: {}'.format(init, mem, k))
            if init == 'normal':
                xq = np.random.randn(1000, D).astype('float32')
                xb = np.random.randn(mem, D).astype('float32')
            else:
                xq = np.random.rand(1000, D).astype('float32')
                xb = np.random.rand(mem, D).astype('float32')
            index.add(xb)
            start = time.time()
            koid = index.search(xq, k)
            print('time: {}'.format(time.time() - start))
            index.reset()