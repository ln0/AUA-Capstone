import numpy as np
from tqdm import tqdm

def VB_GAMP(Y, Omega, nu, eps=1e-05, inpainting=False, max_iter=100):
    
    global S, V
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
    
    O = np.array([np.diag(Omega[:,i]) for i in range(N)])
    mu = np.empty(shape=(M,N,))
    Q = np.empty(shape=(N,M,M)) 
    Set = np.array(np.where(Omega == 1)).T
    L = len(Set)
    
    # params that don't change
    nu_hat = nu + N
    a_hat = L / 2 + a
    
    # initialization of <Sigma> and <gamma>
    gamma_expect = a / b
    Sigma_expect = nu * W
    
    for iter in tqdm(range(max_iter)):
        U, S, V = np.linalg.svd(Sigma_expect) # V == U.T for positive definite Sigma_expect

        # update q_x using GAMP
        for n in range(N):
            mu[:,n], Q[n] = GAMP(k=Y[:,n], pi=Omega[:,n], b=np.zeros(M), xi=gamma_expect)
        
        # update q_sigma
        W_hat = np.linalg.inv(W_inv + mu @ mu.T + sum(Q)) 
        Sigma_expect = nu_hat * W_hat
        
        # update q_gamma
        b_hat = sum([Y[m,n]**2 - 2 * Y[m,n] * mu[m,n] + mu[m,n]**2 + Q[n][m,m] for m, n in Set]) / 2 + b
        gamma_expect = a_hat / b_hat
        
    return mu

def GAMP(k, pi, b, xi, eps=10, max_steps=1):
    
    # Usage: GAMP(k=Y[:,n], pi=Omega[:,n], b=np.zeros(M), xi=gamma_expect)
    
    global S, V
    M = len(b)
    psi = np.zeros(M)
    mu_x = np.array([1 / xi if pi[m]==1 else 0 for m in range(M)])
    phi_x = np.ones(M) * eps
    z = np.empty(shape=(M))
    tao_p = np.empty(shape=(M))
    tao_s = np.empty(shape=(M))
    tao_r = np.empty(shape=(M))
    p = np.empty(shape=(M))
    r = np.empty(shape=(M))

    for step in range(max_steps):

        # Step 1
        z = np.matmul(V, mu_x)
        tao_p = np.matmul(V**2, phi_x)
        p = z-tao_p*psi 
            
        # Step 2
        psi = S * (b - p) / (1 + S * tao_p)
        tao_s = S / (1 + S * tao_p)
        psi = tao_s * (b - p) 
        
        # Step 3
        tao_r = 1 / np.matmul((V**2).T, tao_s)
        r = mu_x + tao_r * np.matmul(V.T, psi)
                 
        # Step 4
        phi_x[pi==0] = tao_r[pi==0]
        mu_x[pi==0] = r[pi==0]
        
        phi_x[pi == 1] = tao_r[pi==1] / (1 + xi * tao_r[pi==1])
        mu_x[pi == 1] = phi_x[pi == 1] * (xi * k[pi == 1] + r[pi==1] / tao_r[pi==1])
        
    return mu_x, np.diag(phi_x)