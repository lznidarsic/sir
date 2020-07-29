#!/usr/bin/env python -W ignore::DeprecationWarning
""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  sysid with uncorellated noise using expectation maximization
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from numpy import zeros, diag, linspace
from src.models import getModel;
from src.Noise import Noise
from src.signals import getSignal
from src.parameterestimation import OEMstep
from src.animations import OPEanimation
from src.Timer import Timer
import matplotlib.pyplot as plt

# System parameters
m   = 0.1;                      # System mass
k   = 2;                        # System spring constant
d   = 0.5;                      # System damping

# Simulation settings
th_r = zeros(3);
th_r[:] = [k/m,d/m,1/m];             # Actual parameter values
th0 = 0.1;

# General Settings
dt  = 1e-2;                     # Time-step
T   = 10.0;                     # Simulation period
fps = 25;

# Noise parameters
zeta = [0.1,0.1,0.2]              # [std(w1),std(w2),std(v)] (mu=0)
seed = [1234];

# Set vectors and dimensions
N = int(1+T/dt)
v = zeros((1,N))
w = zeros((2,N))
u = zeros((1,N))
t = linspace(0,T,N);

# Construct lineat spring-mass damper system and model
S = getModel('SMD');
M = getModel('SMD');
M.th = th_r;
S.th = th_r;
th = zeros(3);

# Construct Gaussian White Noise
pd = zeros((3,2));
pd[:,1] = zeta;

Q = diag(pd[0:2,1]); Q = Q**2;
M.Q = Q; S.Q = Q;

R = zeros((1,1)); R[0,0] = pd[2,1]; R = R**2;
S.R = R; M.R = R;

# Get persistently exciting input signal (sum of sinoids)
u[0,:] = getSignal("LOGSINSUM",t,5,0.1,3,5,1);

# General figure settings
plt.close('all');
hfig = 8;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);

print('Warning: warnings are off')


# Case-specific parameters
alpha = 1e1;
gembed = 40;
M.th = th0*th_r[:];

# set some storage
P = zeros((2,2));
xu = zeros((M.nx,N));
xp = zeros((M.nx,N));
yu = zeros((M.ny,N));
yp = zeros((M.ny,N));
th = zeros((M.nth,N));
M.th = th0*th_r;
th[:,0] = M.th;

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,1,pd,[],[],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

f = plt.figure(1,figsize=(wfig,hfig));
anim = OPEanimation(f,t,x,y,th_r,leg=['Truth','EM']);

# Fetch Timer
Tim = Timer('Animated Online EM',2,N,N);
# Running the online algorithm
for k in range(2,N):
    
    Nk = min(k,gembed);
    d,d,P,d,th[:,-1],xp[:,k],xu[:,k],yp[:,k],yu[:,k] = OEMstep(M,u[:,k-Nk:k+1],\
                                      xu[:,k-Nk:k+1],xp[:,k-Nk:k+1],y[:,k-Nk:k+1],\
                                     yp[:,k-Nk:k+1],th[:,k-Nk:k+1],Q,R,P,alpha)
    M.th = th[:,k];
    
    if k == 2 or k % fps == 0:
            anim.updateAnim(k,xu[:,:k],yu[:,:k],th[:,:k]);
    Tim.looptime(k);


