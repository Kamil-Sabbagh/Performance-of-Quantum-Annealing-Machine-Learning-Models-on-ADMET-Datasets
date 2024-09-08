import neal
import math
from dwave.system import LeapHybridSampler
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
import time
from tqdm import tqdm

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


class SVM:
    def __init__(self,B,K,C,gamma,xi,N,sampler_type) -> None:
        self.gamma = gamma
        self.B = B
        self.K = K 
        self.C = C 
        self.xi = xi 
        self.N = N
        self.sampler_type = sampler_type

        if(sampler_type == 'HQPU'):
            self.sampler = LeapHybridSampler()
        if(sampler_type == 'SA'):
            self.sampler = neal.SimulatedAnnealingSampler()
        if(sampler_type == 'QPU'):
            self.sampler = EmbeddingComposite(DWaveSampler())

        pass

    def kernel(self,x, y):
        if self.gamma == -1:
            k = np.dot(x, y)
        elif self.gamma >= 0:
            k = np.exp(-self.gamma*(np.linalg.norm(x-y, ord=2)))
        
        return k

    def predict_class(self,x_test, alpha, b):
        N = len(alpha)
        f = sum([alpha[n]*self.t[n]*self.kernel(self.data[n], x_test)
                    for n in range(N)]) + b
        return f

    def train_SVM(self,data,t):
        self.data = data 
        self.t = t
        Q_tilde = np.zeros((self.K*self.N, self.K*self.N))
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        Q_tilde[(self.K*n+k, self.K*m+j)] = 0.5*(self.B**(k+j))*t[n]*t[m] * \
                            (self.kernel(data[n], data[m])+self.xi) - \
                            (delta(n, m)*delta(k, j)*(self.B**k))

        Q = np.zeros((self.K*self.N, self.K*self.N))
        for j in range(self.K*self.N):
            Q[(j, j)] = Q_tilde[(j, j)]
            for i in range(self.K*self.N):
                if i < j:
                    Q[(i, j)] = Q_tilde[(i, j)] + Q_tilde[(j, i)]

        size_of_q = Q.shape[0]
        qubo = {(i, j): Q[i, j]
                for i, j in product(range(size_of_q), range(size_of_q))}

        now = time.perf_counter()

        if(self.sampler_type == 'HQPU'):
            response = self.sampler.sample_qubo(qubo)
        if(self.sampler_type == 'SA'):
            response = self.sampler.sample_qubo(qubo, num_reads=100)
        if(self.sampler_type == 'QPU'):
            response = self.sampler.sample_qubo(qubo, num_reads=100)

        print(f'Solver Time: {time.perf_counter() - now}') 

        a = response.first.sample

        alpha = {}
        for n in range(self.N):
            alpha[n] = sum([(self.B**k)*a[self.K*n+k] for k in range(self.K)])

        b = sum([alpha[n]*(self.C-alpha[n])*(t[n]-(sum([alpha[m]*t[m]*self.kernel(data[m], data[n])
                                                    for m in range(self.N)]))) for n in range(self.N)])/sum([alpha[n]*(self.C-alpha[n]) for n in range(self.N)])

        return alpha, b
