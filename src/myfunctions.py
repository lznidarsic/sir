# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:53:36 2020

@author: Laurens
"""

from numpy import dot, transpose, log, zeros, diag, eye
from numpy.linalg import norm, inv

# Multiply multiple matrices using dot-product
def dotn(res,*argv):
    for arg in argv:
        res = dot(res,arg);
    return res;

# Short-shell for matrix-norm logarithm
def ln(A):
    return log(norm(A,2));
    
# Short-shell for transpose;
def tp(res):
    return transpose(res);

# Check if array is empty
def isempty(arr):
    if len(arr) == 0:
        return True;
    else:
        return False;

def isint(var):
    return isinstance(var, int);

# Factorial
def fct(n):
	res = 1;
	for i in range(1,int(n)):
		res*=i
	return res;

# Sum-factorial fsm(1) = 1, fsm(2) = 3, fsm(3) = 6, etc.
def fsm(n):
    return sum(range(1,n+1));

# Array of sum-factorial fsm(1) = 1, fsm(2) = [1,3], fsm(3) = [1,3,6], etc.
def fsma(n):
    res = zeros(n+1);
    for i in range(1,n+1):
        res[i] = res[i-1] + i;
    return res;
   
# Shortcut for (weighted) matrix-square     
def sq(e,W=[]):
    N = len(e[0,:]);
    res=0;
    if isempty(W):
        for i in range(0,N):
            res +=  dot(tp(e[:,i]),  e[:,i]);
    else:
        for i in range(0,N):
            res += dotn(tp(e[:,i]),W,e[:,i]);
    return res;

# Pascal matrix of order k (https://en.wikipedia.org/wiki/Pascal_matrix)  
def pascal(k):
    P = zeros((k,k));
    P[:,0] = 1;
    for i in range(1,k):
        for j in range(1,i+1):
            P[i,j] = P[i-1,j] + P[i-1,j-1];
    return P;

# Diagonal matrix of size k x k with dt^i on the ith row
def powdiag(k,a):
    D = [];
    for i in range(0,k):
        D.append(a**i),
    return diag(D);
    
# Numerical gradient
def NG(y,x,d=1e-6):
    
    nx = len(x);
    dx = zeros(nx);
    
    for i in range(0,nx):
        
        # Backwards-step value
        x[i] += - d;
        Jm    =   y(x)[0];
        
        # Forwards-step value
        x[i] += 2*d;
        Jp    =   y(x)[0];
        
        # Get gradient dy/dxi
        dx[i] = (Jp - Jm)/(2*d);
        
        # Restore x
        x[i] += - d;
        
    return dx;  
    
def dare(A,C,Q,R,tol=1e-6,maxit=1000,convthresh=200):
    
    nx = len(A[:,0]);
    ny = len(C[:,0]);
    
    P = zeros((nx,nx));
    S = zeros((ny,ny));
    K = zeros((nx,ny));
    e1 = 1000;
    J = 1000;
    e2 = 0;
    i = 0;
    j = 0;
    
    while i < maxit and j < convthresh:
        
        S = dotn(C,P,tp(C)) + R;
        P = dotn(A,P,tp(A)) + Q;
        K = dotn(P,tp(C),inv(S));
        P = dotn(eye(2)-dotn(K,C),P);
        
        
        # Stopping criterion: J > tol in last convthresh iterations
        
        J = P - dotn(A,P,tp(A)) + Q;
        
        
        
        
        e2 = e1;
        e1 = norm(K,2) + norm(P,2)  + norm(S,2);
        J = abs(e2-e1);
        if j == 0:
            if J < tol:
                j = 1;
        else:
            if J < tol:
                j +=1;
            else:
                j = 0;
        
        i +=1;
    
    return K,P,S;