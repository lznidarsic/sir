""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  Free-Energy filter tested in three different settings
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from numpy import zeros, diag
from src.models import getModel;
from src.Noise import Noise
from src.signals import getSignal
from src.parameterestimation import DTCM, TCM, generalize, FEF, NCNDO, KF
from src.visualizations import plotSimulation, tudcols
from src.filtering import dare
import matplotlib.pyplot as plt

# System parameters
m   = 0.1;                      # System mass
k   = 2;                        # System spring constant
d   = 0.5;                      # System damping

# Simulation settings
th = [k/m,d/m,1/m];             # Actual parameter values
maxth = 10;                     # Maximum deviation of actual param value
res = 51;                       # Resolution, number of values to be eval'd
dth = 2*maxth/(res-1);          # Parameter step-size;

# General Settings
dt  = 1e-2;                     # Time-step
T   = 10.0;                     # Simulation period

# DEM Settings
p       = 6;                    # Embedding order
s       = 1;                    # Kernel width (in # of time-steps)

# Noise parameters
zeta = [0.1,0.1,0.2]              # [std(w1),std(w2),std(v)] (mu=0)
seed = [1234];

# Set vectors and dimensions
N = int(1+T/dt)
v = zeros((1,N))
w = zeros((2,N))
u = zeros((1,N))

# Construct lineat spring-mass damper system and model
S = getModel('SMD');
M = getModel('SMD');
M.th = th;
S.th = th;

# Construct Gaussian White Noise
pd = zeros((3,2));
pd[:,1] = zeta;
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
t,w_tmp = W.getNoise();

# Get temporal corellation matrix
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];

Q = diag(pd[0:2,1]); Q = Q**2;
M.Q = Q; S.Q = Q;

R = zeros((1,1)); R[0,0] = pd[2,1]; R = R**2;
S.R = R; M.R = R;

# Get persistently exciting input signal (sum of sinoids)
u[0,:] = getSignal("LOGSINSUM",t,5,0.1,3,5,1);

# Simulate system (generate training data)
x,y     = S.sim([0,0],u,w,v);
x0 = zeros(p*M.nx);

# Generalize Model
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];

# General figure settings
plt.close('all');
hfig = 4;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);

#%%  Case 1: embedded predictions, kernel with = 0.01s;

# Temporal corellation
Sc = DTCM('GF',p,s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-3              

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,0);

# Plot
f1 = plt.figure(1,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded predictions'],loc=1);
plt.savefig('GFEP001.png', di=300);

#%%  Case 2: embedded derivatives, kernel with = 0.01s;

# Temporal corellation
Sc = TCM('GF',p,dt*s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-2                 

# Generalize in- and output signals;
u_t = NCNDO(p,u,dt);
y_t = NCNDO(p,y,dt);

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,1);

# Plot
f2 = plt.figure(2,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded derivatives'],loc=1);
plt.savefig('GFED001.png', di=300);


#%%  Case 3: embedded history, kernel with = 0.01s;

# Temporal corellation
Sc = DTCM('GF',p,s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-3                 

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,2);

# Plot
f3 = plt.figure(3,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded history'],loc=1);
plt.savefig('GFEH001.png', di=300);


#%%  Case 4: Kalman Filter, kernel with = 0.01s;

# Obtain data
xp,xu,yp,yu = KF(M,[0,0],u,y)[3:7];

# plot
f4 = plt.figure(4,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu,yu,[],[],tudcols[0],False);
plt.legend(['Truth','KF'],loc=1);
plt.savefig('KF001.png', di=300);

#%% Intermezzo: reset noise and re-simulate system

# DEM Settings
s       = 10;                    # Kernel width (in # of time-steps)

W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
t,w_tmp = W.getNoise();
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];

# Simulate system (generate training data)
x,y     = S.sim([0,0],u,w,v);

#%%  Case 5: embedded predictions, kernel with = 0.1s;

# Temporal corellation
Sc = DTCM('GF',p,s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-3                 

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,0);

# Plot
f5 = plt.figure(5,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded predictions'],loc=1);
plt.savefig('GFEP01.png', di=300);

#%%  Case 6: embedded derivatives, kernel with = 0.1s;

# Temporal corellation
Sc = TCM('GF',p,dt*s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-2                 

# Generalize in- and output signals;
u_t = NCNDO(p,u,dt);
y_t = NCNDO(p,y,dt);

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,1);

# Plot
f6 = plt.figure(6,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded derivatives'],loc=1);
plt.savefig('GFED01.png', di=300);


#%%  Case 7: embedded history, kernel with = 0.1s;

# Temporal corellation
Sc = DTCM('GF',p,s);

# Generalize Model
M_t = generalize(M,p,Sc)[0];

# State-estimation gain
alpha   = 5e-3                 

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Obtain data
xu_t,xp_t,yu_t,yp_t = FEF(M_t,x0,u_t,y_t,p,alpha,2);

# Plot
f7 = plt.figure(7,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu_t[0:2,:],yu_t[0:1,:],[],[],tudcols[0],False);
plt.legend(['Truth','GF with Embedded history'],loc=1);
plt.savefig('GFEH01.png', di=300);


#%%  Case 8: Kalman Filter, kernel with = 0.1s;

# Obtain data
xp,xu,yp,yu = KF(M,[0,0],u,y)[3:7];

# plot
f8 = plt.figure(8,figsize=(wfig,hfig));
plotSimulation(t,[],x,y,[],[],'black');
plotSimulation(t,[],xu,yu,[],[],tudcols[0],False);
plt.legend(['Truth','KF'],loc=1);
plt.savefig('KF01.png', di=300);

