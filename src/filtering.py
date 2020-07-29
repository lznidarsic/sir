# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:34:14 2020

@author: Laurens
"""
from numpy import zeros, eye
from numpy.linalg import norm, inv
from src.myfunctions import dotn, tp

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

# xp,xu,yp,yu = sgfilter(A,B,C,D,K,x0,u,y);
def sgfilter(A,B,C,D,K,x0,u,y):
    
    nx = len(A[:,0]);
    ny = len(C[:,0]);
    N =  len(u[0,:]);
    
    xp = zeros((nx,N)); xp[:,0] = x0;
    xu = zeros((nx,N)); xu[:,0] = x0;
    yp = zeros((ny,N)); yp[:,0] = dotn(C,xp[:,0]) + dotn(D,u[:,0]);
    yu = zeros((ny,N)); yu[:,0] = dotn(C,xu[:,0]) + dotn(D,u[:,0]);
    
    for k in range(1,N):
        
        # Predict
        xp[:,k] = dotn(A,xu[:,k-1]) + dotn(B,u[:,k-1]);
        yp[:,k] = dotn(C,xp[:,k  ]) + dotn(D,u[:,k  ]);
        
        # Update
        xu[:,k] =        xp[:,k]    + dotn(K,y[:,k] - yp[:,k]);
        yu[:,k] = dotn(C,xp[:,k  ]) + dotn(D,u[:,k]          );
        
    return xp,xu,yp,yu;