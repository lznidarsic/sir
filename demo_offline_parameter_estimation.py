#!/usr/bin/env python -W ignore::DeprecationWarning
""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  sysid with uncorellated noise using expectation maximization
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from numpy import zeros, diag, dot, eye, linspace
from scipy.linalg import expm, kron
from src.models import getModel;
from src.Noise import Noise
from src.signals import getSignal
from src.parameterestimation import qcf, DTCM, TCM, generalize, ll, NCNDO,KF,FEF,EM,DEM
from src.visualizations import plotP, tudcols
from src.Timer import Timer
from src.myfunctions import dare
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
maxit = 1000;

# DEM Settings
p       = 6;                    # Embedding order

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
u = zeros((1,N));
u[0,:] = getSignal("LOGSINSUM",t,5,0.1,3,5,1);


# Some storage
xu_t = zeros((2*p,N));
xp_t = zeros((2*p,N));
yp_t = zeros((1*p,N));

# General figure settings
plt.close('all');
hfig = 6;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);

print('Warning: warnings are off')

#%% Case 1.1: EM with kernel width = 0.01 s

# Case-specific parameters
s = 1;
alpha = 2e-3;         
gembed = 20
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,1,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

## Running the algorithm
dat = EM(M,[0,0],u,y,Q,R,alpha,maxit,False,1e-6,100,gembed);

# Plot parameter estimation
f1 = plt.figure(1,figsize=(wfig,hfig));
plotP(dat,th_r,tudcols[0],True,nrows=2,irow=1);
plotP(dat,th_r,tudcols[0],False,nrows=2,irow=1);

#%% Case 1.2: EM with kernel width = 0.1 s

# Case-specific parameters
s = 10;
alpha = 5e-4;         
gembed = 20;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

## Running the algorithm
dat = EM(M,[0,0],u,y,Q,R,alpha,maxit,False,1e-6,100,gembed);
 
# Plot parameter estimation
plotP(dat,th_r,tudcols[0],True,nrows=2,irow=2);
plotP(dat,th_r,tudcols[0],False,nrows=2,irow=2);
plt.legend(['Truth','EM'],loc=4);
plt.savefig('EM.png', dpi=300);

#%% Case 2.1: DEM with embedded predictions and kernel width = 0.01 s

# Case-specific parameters
s = 1;
alpha_x = 5e-3;
alpha_th = 2e-5;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (predictions)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    x_t[M.nx*i:M.nx*(i+1),0:N-i] = x[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,0,gembed,maxit,False,False);

# Plot parameter estimation
f2 = plt.figure(2,figsize=(wfig,hfig));
plotP(dat,th_r,tudcols[0],True,2,1);
plotP(dat,th_r,tudcols[0],False,2,1);

#%% Case 2.2: DEM with embedded predictions and kernel width = 0.1 s

# Case-specific parameters
s = 10;
alpha_x = 5e-3;
alpha_th = 1e-6;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (predictions)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    x_t[M.nx*i:M.nx*(i+1),0:N-i] = x[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,0,gembed,maxit,False,False);

# Plot parameter estimation
plotP(dat,th_r,tudcols[0],True,2,2);
plotP(dat,th_r,tudcols[0],False,2,2);
plt.legend(['Truth','DEM with generalized predictions'],loc=4);
plt.savefig('DEMGP.png', dpi=300);

#%% Case 3.1: DEM with embedded derivatives and kernel width = 0.01 s

# Case-specific parameters
s = 1;
alpha_x = 5e-3;
alpha_th = 2e-5;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,1,gembed,maxit,False,False);

# Plot parameter estimation
f3 = plt.figure(3,figsize=(wfig,hfig));
plotP(dat,th_r,tudcols[0],True,2,1);
plotP(dat,th_r,tudcols[0],False,2,1);

#%% Case 3.2: DEM with embedded derivatives and kernel width = 0.1 s

# Case-specific parameters
s = 10;
alpha_x = 5e-3;
alpha_th = 1e-6;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,1,gembed,maxit,False,False);

# Plot parameter estimation
plotP(dat,th_r,tudcols[0],True,2,2);
plotP(dat,th_r,tudcols[0],False,2,2);
plt.legend(['Truth','DEM with generalized derivatives'],loc=4);
plt.savefig('DEMGD.png', dpi=300);

#%% Case 4.1: DEM with embedded history and kernel width = 0.01 s

# Case-specific parameters
s = 1;
alpha_x = 5e-3;
alpha_th = 2e-5;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,1,gembed,maxit,False,False);

# Plot parameter estimation
f4 = plt.figure(4,figsize=(wfig,hfig));
plotP(dat,th_r,tudcols[0],True,2,1);
plotP(dat,th_r,tudcols[0],False,2,1);

#%% Case 4.2: DEM with embedded history and kernel width = 0.1 s

# Case-specific parameters
s = 10;
alpha_x = 5e-3;
alpha_th = 1e-6;
gembed = 1;
M.th = th0*th_r[:];

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Running the algorithm
dat = DEM(M_t,zeros(p*M.nx),u_t,y_t,p,M_t.Q,M_t.R,alpha_x,alpha_th,1,gembed,maxit,False,False);

# Plot parameter estimation
plotP(dat,th_r,tudcols[0],True,2,2);
plotP(dat,th_r,tudcols[0],False,2,2);
plt.legend(['Truth','DEM with generalized history'],loc=4);
plt.savefig('DEMGH.png', dpi=300);
