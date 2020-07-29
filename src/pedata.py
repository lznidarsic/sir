# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:04:41 2020

@author: Laurens
"""

from numpy import linspace;

class pedata:
    """
    Initializer / Instance Attributes
    algo   name of algorithm
    u      input
    xp     predicted hidden state
    xc     corrected hidden state
    yp     predicted output
    p      estimated parameters
    Q      estimated idden state covariance
    R      estimated output covariance
    J      cost function
    """
    def __init__(self,algo='nameless',M=[],J=[],K=[],P=[],S=[],th=[],xp=[],xu=[],yp=[],yu=[]):
        self.algo=algo;
        self.M = M;
        self.it = linspace(1,len(J[0,:]),len(J[0,:]));
        self.J = J;
        self.K = K;
        self.P = P;
        self.S = S;
        self.th = th;
        self.t = linspace(0,M.dt*(len(xp[0,:])-1),len(xp[0,:]));
        self.xp = xp;
        self.xu = xu;
        self.yp = yp;
        self.yu = yu;


        