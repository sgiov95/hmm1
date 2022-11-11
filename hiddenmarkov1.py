"""
Created on Thu Aug 18 13:11:19 2022

@author: Sebastian

hiddenmarkov1.py

This is a script that focuses on functions that one would use to analyze 
a hidden markov model

General functions include:
-Stable Distribution
-Joint Distribution
-Inverse Condition Probability
-Inverse Transition Probability
-Joint Sample Likelihood

Inferential functions include:
-Forward Probabilities
-Backward Probabilities
-Gamma
-Zeta
-Viterbi Path
-Baum Welch

Other functions include:
-Full Markov Sample Estimate
-Minerbi Path

"""

#libraries
import numpy as np
import matplotlib.pyplot as plt
import re

#general functions

'''
    Stable Distribution
    
Inputs:
    Square Row Stochastic Matrix A (size m by m)
    
Outputs:
    A m dimensional Stochastic Vector p
    
'''
def Stable_Distribution(A,r=1000):
    u = np.mean(np.linalg.matrix_power(A,r),axis=0)
    u /= sum(u)
    return u
'''
    Joint Distribution
    
Inputs: 
    A Row Stochastic Matrix A (size m by k)
    A m dimensional Column Vector c 
    
Outputs:
   A Joint Probability table the same size as A
'''
def Joint_Distribution(A,c):
    m = A.shape[0]
    k = A.shape[1]
    C = np.tile(c,k).reshape((k,m)).transpose()
    B = A*C
    return B
'''
    Inverse Condition Probability
    
Inputs: 
    A Row Stochastic Matrix A (m,m)
    A Row Stochastic Matrix B (m,k)
    Optional: A m-dimensional vector c
    If None is chosen, we will just use the
    Stable Distribution
    
Outputs:
    A Row Stochastic Matrix Q (k,m)
'''
def Inverse_Conditional_Probability(A,B,c=None):
    m = A.shape[0]
    k = B.shape[1]
    if c is None:
        c = Stable_Distribution(A)
    P = Joint_Distribution(B,c)
    q = P.sum(0)
    Q = P.transpose()
    Q /= np.tile(q,m).reshape((m,k)).transpose()
    return Q
'''
    Inverse Transition Probability
    
Inputs:
    A Row Stochastic Matrix A (m,m)
    A Row Stochastic Matrix B (m,k)
    Optional: A m-dimensional vector c
    If None is chosen, we will just use the
    Stable Distribution
    Optional: A order r, preset to 1,
    representing how far backwards the
    transition is

Outputs:
    A k by m Row Stochastic Matrix
    Representing P(X[t+r]|Y[t])        
'''
def Inverse_Transition_Probability(A,B,c=None,r=1):
    m = A.shape[0]
    k = B.shape[1]
    P = Inverse_Conditional_Probability(A,B,c)
    R = np.linalg.matrix_power(A,r)
    Q = np.matmul(R,P.transpose())
    return Q
'''
    Joint Sample Likelihood
    
Inputs:
    A Row Stochastic Matrix A (m,m)
    A Row Stochastic Matrix B (m,k)
    Optional: A m-dimensional vector c
    If None is chosen, we will just use the
    Stable Distribution
    Sequences x (hidden) and y (observed)

Output:
    A probability of the sample    
'''
def Joint_Sample_Likelihood(x,y,A,B,c=None):
    n = len(x)
    m = A.shape[0]
    k = B.shape[1]
    if c is None:
        c = Stable_Distribution(A)
    #initial state
    p = np.zeros(n)
    p[0] = B[x[0],y[0]]*c[x[0]]
    #iterations:
    for i in range(1,n):
        p[i] = B[x[i],y[i]]*A[x[i-1],x[i]]*p[i-1]
    return p
#inferential functions
'''
    Fwd Probabilities
    
Inputs: A,B,c
        Y
    
Outputs:
        alpha vector at times t=0,1,..,N-1
        for each sequence in the set
'''

def alphap(Y,A,B,c=None):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    if c is None:
        c = Stable_Distribution(A)
    fpr = list()
    for i in range(s):
        temp = np.zeros((slen[i],m))
        temp[0,:] = c*B[:,Y[i][0]]
        for t in range(1,slen[i]):
            temp1 = np.zeros(m)
            for j in range(m):
                temp1 += (temp[t-1,j]*A[j,:])
            temp[t,:] = B[:,Y[i][t]]*temp1
        fpr.append(temp)
    return fpr
'''
    Bwd Probabilities
    
Inputs: A,B
        Y
        
Outputs:
        beta vector at times t=0,1,...,N-1
        for each sequence in the set
'''
def betap(Y,A,B):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    bpr = list()
    for i in range(s):
        temp = np.ones((slen[i],m))
        for t in range(1,slen[i]):
            temp1 = np.zeros(m)
            for j in range(m):
                temp1 += (B[j,Y[i][slen[i]-t]]*temp[slen[i]-t,:]*A[:,j])
            temp[slen[i]-1-t,:] = temp1
        bpr.append(temp)
    return bpr
'''
    Gamma 
    
Inputs: alpha, beta

Outputs:
        gamma vector at times t=0,1,...,N-1
        for each sequence in the set
'''
def gammap(A,B,alpha,beta,Y):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    gpr = list()
    for i in range(s):
        temp = np.ones((slen[i],m))
        for t in range(slen[i]):
            temp1 = np.zeros(m)
            for j in range(m):
                temp1[j] = alpha[i][t,j]*beta[i][t,j]
            if sum(temp1) > 0:
                temp1 /= sum(temp1)
            temp[t,:] = temp1
        gpr.append(temp)
    return gpr
    
'''
    Zeta
    
Inputs: A,B
        Y
        alpha, beta
    
Outputs:
        zeta matrix at times t=0,1,...,N-2
'''
def zetap(A,B,alpha,beta,Y):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    zpr = list()
    for i in range(s):
        temp = np.ones((slen[i]-1,m,m))
        for t in range(slen[i]-1):
            temp1 = np.zeros((m,m))
            for u in range(m):
                for v in range(m):
                    temp1[u,v] = alpha[i][t,u]*A[u,v]*beta[i][t+1,v]*B[v,Y[i][t+1]]
            if np.sum(temp1) > 0:
                temp1 /= np.sum(temp1)
            temp[t,:] = temp1
        zpr.append(temp)
    return zpr
'''
    Viterbi Path
    
Inputs:
        A,B,c
        Y
    
Outputs:
        X, set of hidden sequence estimates that
        corr. to Y, the observed sample set
'''
def Viterbi_Path(Y,A,B,c=None):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    if c is None:
        c = Stable_Distribution(A)
    X = list()
    for i in range(s):
        x = np.zeros(slen[i],dtype=int)
        y = Y[i]
        p = Joint_Distribution(B,c)[:,y[0]]
        x[0] = np.argmax(p)
        for t in range(1,slen[i]):
            p = p[x[t-1]]*(B[:,y[t]]*A[x[t-1],:])
            x[t] = np.argmax(p)
        X.append(x)
    return X
        
'''
    Baum Welch Algorithm
    
Inputs:
    A r0w-stochastic matrix A (m,m)
    A row stochastic matrix B (m,k)
    optional initial probability c (m)
    if None, then use Stable Distribution
    for A
    Collection of sequences Y as a list of 
    numpy int arrays
    r = 1 default, number of times to run through
    algorithm, optional
    
Outputs:
    New estimates for A,B,c that are row-stochastic    
'''
def Baum_Welch(Y,A,B,c=None,r=1):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    if c is None:
        c = Stable_Distribution(A)
    for i in range(r):
        alpha = alphap(Y,A,B,c)
        beta = betap(Y,A,B)
        gamma = gammap(A,B,alpha,beta,Y)
        zeta = zetap(A,B,alpha,beta,Y)
        ctemp = np.zeros(m)
        for u in range(s):
            ctemp += gamma[u][0,:]
        c = ctemp/s
        Atemp1 = np.zeros((m,m))
        Atemp2 = np.zeros(m)
        for u in range(s):
            for v in range(slen[u]-1):
                Atemp1 += zeta[u][v,:,:]
                Atemp2 += gamma[u][v,:]
        for u in range(m):
            for v in range(m):
                A[u,v] = Atemp1[u,v]/Atemp2[u]
        Btemp1 = np.zeros((m,k))
        Btemp2 = np.zeros(m)
        for u in range(s):
            for v in range(slen[u]):
                for w in range(m):
                    for j in range(k):
                        if Y[u][v] == j:
                            Btemp1[w,j] += gamma[u][v,w]
                    Btemp2[w] += gamma[u][v,w]
        for u in range(m):
            for v in range(k):
                B[u,v] = Btemp1[u,v]/Btemp2[u]
        a = A.sum(1)
        A /= np.tile(a,m).reshape(A.shape).transpose()
        b = B.sum(1)
        B /= np.tile(b,k).reshape((k,m)).transpose()
        c /= sum(c)
    return A,B,c
            

#other functions
'''
    Weighted Distribution Weighing
    
Inputs: 
    A matrix type object that is called A (m,k)
    Optional matrix of the same size
    Noise factor, f > 0, if none, then = 1
    
Outputs:
    A matrix of the same dimensions as A that is row
    stochastic
'''
def Fuse_Distributions(A,B=None,f=1):
    m = A.shape[0]; k = A.shape[1]
    if B is None:
        B = np.random.uniform((m,k))
    Q = A + f*B
    c = Q.sum(1)
    Q /= np.tile(c,k).reshape((k,m)).transpose()
    return Q

'''
    Markov Full Sample Estimate
    
Inputs: x and y,

Output: Matrices A, B, c
'''
def Sample_Estimate(X,Y,m,k):
    s = len(Y)
    A = np.zeros((m,m))
    B = np.zeros((m,k))
    c = np.zeros(m)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    for r in range(s):
        for i in range(m):
            if X[r][0] == i:
                c[i] += 1
    c /= s
    for r in range(s):
        ns = slen[r]
        for t in range(1,ns):
            for i in range(m):
                for j in range(m):
                    if (X[r][t-1] == i) and (X[r][t] == j):
                        A[i,j] += 1
    a = A.sum(1)
    A /= np.tile(a,m).reshape((m,m)).transpose()
    for r in range(s):
        ns = slen[r]
        for t in range(ns):
            for i in range(m):
                for j in range(k):
                    if (X[r][t] == i) and (Y[r][t] == j):
                        B[i,j] += 1
    b = B.sum(1)
    B /= np.tile(b,k).reshape((k,m)).transpose()
    return A,B,c
'''
    Discrete Time Markov Chain
    
Inputs: 
    
Outputs:
'''
def dtmc(n,A,c,x0=None):
    m = A.shape[0]
    x = np.zeros(n,dtype=int)
    if x0 is None:
        prob = c
        x0 = np.random.choice(m,size=1,p=prob)
    x[0] = int(x0)
    prev = x[0]
    for i in range(1,n):
        prob = A[prev,:]
        x[i] = np.random.choice(m,size=1,p=prob)
        prev = x[i]
    return x
'''
    Hidden Markov Chain
    
Inputs:
    
Outputs:
'''
def hmc(n,A,B,c):
    m = A.shape[0]
    k = B.shape[1]
    x = dtmc(n,A,c)
    y = np.zeros(n,dtype=int)
    for i in range(n):
        y[i] = np.random.choice(k,size=1,p=B[x[i],])
    return x,y
        
'''
    Minerbi Path
    
Inputs: A,B,c
        Y
    
Outputs:
        Z, sequence set that best avoids X based on the
        information from Y
'''
def Minerbi_Path(Y,A,B,c=None):
    m = A.shape[0]; k = B.shape[1]
    s = len(Y)
    slen = np.zeros(s,dtype=int)
    for i in range(s):
        slen[i] = len(Y[i])
    if c is None:
        c = Stable_Distribution(A)
    X = Viterbi_Path(Y,A,B,c)
    Z = list()
    for i in range(s):
        z = np.zeros(slen[i],dtype=int)
        y = Y[i]
        x = X[i]
        p = Joint_Distribution(B,c)[:,y[0]]
        z[0] = np.argmin(p)
        for t in range(1,slen[i]):
            p /= sum(p)
            p = p[x[t-1]]*(B[:,y[t]]*A[x[t-1],:])
            z[t] = np.argmin(p)
        Z.append(z)
    return Z    
'''
    Encoder 
    
Inputs:
        x, a vector of n values
        m, the number of states in x
        r, the order of the markov chain
'''
def ncode(x,m,r):
    n = len(x)
    q = n-r+1
    z = np.zeros(q,dtype=int)
    ov = 0
    for i in range(r-1):
        ov = m*ov + x[r-i-2]
    z[0] = (m**(r-1))*x[r-1]+ov
    for i in range(1,q):
        ov = int((z[i-1] - x[i-1])/m)
        z[i] = (m**(r-1))*x[i+r-1]+ov
    return z
'''
    Decoder
    
Inputs:
        z, a vector of q values
        m, the number of states in x
        r, the order of the markov chain
'''
def dcode(z,m,r):
    q = len(z)
    n = q+r-1
    x = np.zeros(n,dtype=int)
    a = z[0]
    for i in range(r):
        x[i] = a%m
        a = int(a/m)
    for i in range(1,q):
        x[i+r-1] = (z[i]/(m**(r-1)))%m
    return x

