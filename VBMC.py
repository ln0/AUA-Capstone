import numpy as np
from tqdm import tqdm

def VBMC(Y, Omega, nu, eps=1e-05, inpainting=False, max_iter=10000):

    M, N = Y.shape
    a, b = 1e-08, 1e-08
    W = eps**-1 * np.eye(M)

    if inpainting:
        F = np.empty(shape=(M,M,))
        for i in range(M):
            for j in range(M):
                if i == j:
                    F[i,j] = -2
                elif abs(i - j) == 1:
                    F[i,j] = 1
                else: 
                    F[i,j] = 0
        W = F.T @ F

    W_inv = np.linalg.inv(W)
    
    O = [np.diag(Omega[:,i]) for i in range(N)]
    mu = np.empty(shape=(M,N,))
    Q = np.empty(shape=(N,M,M))
    S = np.array(np.where(Omega == 1)).T
    L = len(S)
    
    # params that don't change
    nu_hat = nu + N
    a_hat = L / 2 + a
    
    # initialization of <Sigma> and <gamma>
    gamma_expect = a / b
    Sigma_expect = nu * W
    
    for iter in tqdm(range(max_iter)):
        for n in range(N):
            
            # update q_x
            Q[n] = np.linalg.inv(gamma_expect * O[n] + Sigma_expect)
            mu[:,n] = gamma_expect * Q[n] @ O[n] @ Y[:,n]
            
        # update q_sigma
        W_hat = np.linalg.inv(W_inv + mu @ mu.T + sum(Q)) 
        Sigma_expect = nu_hat * W_hat
        
        # update q_gamma
        b_hat = sum([Y[m,n]**2 - 2 * Y[m,n] * mu[m,n] + mu[m,n]**2 + Q[n][m,m] for m, n in S]) / 2 + b
        gamma_expect = a_hat / b_hat
        
    return mu