import math

import numpy as np


class FairRecAlg:
    def __init__(self, rec_size=10, MMS_fraction=0.5):
        self.rec_size = rec_size
        self.MMS_fraction = MMS_fraction

    def fit(self, X):
        self.m = X.shape[0]
        self.n = X.shape[1]
        U = range(self.m)
        P = range(self.n)
        A = {}
        for u in U:
            A[u] = []
        F = {}
        for u in U:
            F[u] = P[:]
        l = int(self.MMS_fraction * self.m * self.rec_size / (self.n + 0.0))
        R = int(math.ceil((l * self.n) / (self.m + 0.0)))
        T = l * self.n
        [B, F1] = self._greedy_round_robin(R, l, T, X, U[:], F.copy())
        F = {}
        F = F1.copy()
        print("GRR done")
        for u in U:
            A[u] = A[u][:] + B[u][:]
        u_less = []
        for u in A:
            if len(A[u]) < self.rec_size:
                u_less.append(u)
        for u in u_less:
            scores = X[u, :]
            new = scores.argsort()[-(self.rec_size + self.rec_size) :][::-1]
            for p in new:
                if p not in A[u]:
                    A[u].append(p)
                if len(A[u]) == self.rec_size:
                    break

        return A

    def _greedy_round_robin(self, R, l, T, V, U, F):
        """greedy round robin allocation based on a specific ordering of
        customers (assuming the ordering is done in the relevance scoring
        matrix before passing it here)"""

        B = {}
        for u in U:
            B[u] = []
        Z = {}
        P = range(self.n)
        for p in P:
            Z[p] = l
        for t in range(1, R + 1):
            print("GRR round number: ", t)
            for i in range(self.m):
                if T == 0:
                    return B, F
                u = U[i]
                possible = [(Z[p] > 0) * (p in F[u]) * V[u, p] for p in range(self.n)]
                p_ = np.argmax(possible)
                if (Z[p_] > 0) and (p_ in F[u]) and len(F[u]) > 0:
                    F[u] = list(F[u])
                    B[u].append(p_)
                    F[u].remove(p_)
                    Z[p_] = Z[p_] - 1
                    T = T - 1
                else:
                    return B, F
        return B, F
