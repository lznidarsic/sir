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
from src.parameterestimation import qcf, DTCM, TCM, generalize, ll, NCNDO,KF,FEF
from src.visualizations import plotcost, tudcols
from src.Timer import Timer
from src.myfunctions import dare
import matplotlib.pyplot as plt

# System parameters
m   = 0.1;                      # System mass
k   = 2;                        # System spring constant
d   = 0.5;                      # System damping

# Simulation settings
th_r = [k/m,d/m,1/m];             # Actual parameter values

# General Settings
dt  = 1e-2;                     # Time-step
T   = 10.0;                     # Simulation period
res = 51;                       # Resolution, number of values to be eval'd

# DEM Settings
alpha   = 5e-3                  # State-estimation gain
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

#%% Case 1.1: EM with known states, kernel width = 0.01 s

# Case-specific parameters
maxth = 4;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Fetch Timer
Tim = Timer('Running cost estimates for case 1/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M.th   = th;
        xu   = zeros((M.nx,N));
        xp   = zeros((M.nx,N));
        yp   = zeros((M.ny,N));
        
        # Log-likelihood
        K = dare(M.A([0,0],[0],th),M.C([0,0],[0],th),Q,R)[0];
        
        for k in range(1,N):
            xp[:,k] = M.f(x[:,k-1], u[:,k-1], th);
            yp[:,k] = M.h(xp[:,k], u[:,k], th);
            xu[:,k] = x[:,k] + dot(K,y[:,k]-yp[:,k]);
            
        # Calculate cost
        Jt[i,j] = -ll(y,yp,R,x,xp,Q);
        Jp[i,j] = -ll(y,yp,R,xu,xp,Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f1 = plt.figure(1,figsize=(wfig,hfig));
plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 1.2: EM with known states, kernel width = 0.1 s

# Case-specific parameters
maxth = 4;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Fetch Timer
Tim = Timer('Running cost estimates for case 2/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M.th   = th;
        xu   = zeros((M.nx,N));
        xp   = zeros((M.nx,N));
        yp   = zeros((M.ny,N));
        
        # Log-likelihood
        K = dare(M.A([0,0],[0],th),M.C([0,0],[0],th),Q,R)[0];
        
        for k in range(1,N):
            xp[:,k] = M.f(x[:,k-1], u[:,k-1], th);
            yp[:,k] = M.h(xp[:,k], u[:,k], th);
            xu[:,k] = x[:,k] + dot(K,y[:,k]-yp[:,k]);
            
        # Calculate cost
        Jt[i,j] = -ll(y,yp,R,x,xp,Q);
        Jp[i,j] = -ll(y,yp,R,xu,xp,Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Theoretical cost','Practical cost'],loc=10);
plt.savefig('KS_EM.png', dpi=300);

#%% Case 1.3: EM with unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 4;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Fetch Timer
Tim = Timer('Running cost estimates for case 3/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M.th   = th;
        xp,xu,yp,yu = KF(M,[0,0],u,y)[3:7];
            
        # Calculate cost
        #Jt[i,j] = -ll(y,yp,R,x,xp,Q);
        Jp[i,j] = -ll(y,yp,R,xu,xp,Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f2 = plt.figure(2,figsize=(wfig,hfig));
#plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 1.4: EM with unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 4;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Fetch Timer
Tim = Timer('Running cost estimates for case 4/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M.th   = th;
        xp,xu,yp,yu = KF(M,[0,0],u,y)[3:7];
            
        # Calculate cost
        #Jt[i,j] = -ll(y,yp,R,x,xp,Q);
        Jp[i,j] = -ll(y,yp,R,xu,xp,Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    

#plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Cost'],loc=10);
plt.savefig('US_EM.png', dpi=300);

#%% Case 2.1: DEM with embedded predictions and known states, kernel width = 0.01 s


# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,1),eye(2));   

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (predictions)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    x_t[M.nx*i:M.nx*(i+1),0:N-i] = x[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Fetch Timer
Tim = Timer('Running cost estimates for case 5/16',0,3*res,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f3 = plt.figure(3,figsize=(wfig,hfig));
plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 2.2: DEM with embedded predictions and known states, kernel width = 0.1 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;
alpha = 5e-3;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,1),eye(2));   

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (predictions)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),0:N-i] = u[:,i:N];
    x_t[M.nx*i:M.nx*(i+1),0:N-i] = x[:,i:N];
    y_t[M.ny*i:M.ny*(i+1),0:N-i] = y[:,i:N];

# Fetch Timer
Tim = Timer('Running cost estimates for case 6/16',0,3*res,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Theoretical cost','Practical cost'],loc=10);
plt.savefig('KS_DEMGP.png', dpi=300);

#%% Case 2.3: DEM with embedded predictions and unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

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

# Fetch Timer
Tim = Timer('Running cost estimates for case 7/16',0,3*res,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,0)
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f4 = plt.figure(4,figsize=(wfig,hfig));
#plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 2.4: DEM with embedded predictions and unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

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

# Fetch Timer
Tim = Timer('Running cost estimates for case 8/16',0,3*res,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,0)
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
#plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Cost'],loc=10);
plt.savefig('US_DEMGP.png', dpi=300);

#%% Case 3.1: DEM with embedded derivatives and known states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];
D = expm(D*dt);

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];

# Fetch Timer
Tim = Timer('Running cost estimates for case 9/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f5 = plt.figure(5,figsize=(wfig,hfig));
plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 3.2: DEM with embedded derivatives and known states, kernel width = 0.1 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];
D = expm(D*dt);

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];

# Fetch Timer
Tim = Timer('Running cost estimates for case 10/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Theoretical cost','Practical cost'],loc=10);
plt.savefig('KS_DEMGD.png', dpi=300);

#%% Case 3.3: DEM with embedded derivatives and unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];
D = expm(D*dt);

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];
    
# Fetch Timer
Tim = Timer('Running cost estimates for case 11/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,1);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f6 = plt.figure(6,figsize=(wfig,hfig));
#plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 3.4: DEM with embedded derivatives and unknown states, kernel width = 0.1 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,TCM('GF',p,s*dt))[0];
D = expm(D*dt);

# Generalized signals (derivative)
x_t = zeros((p*M.nx,N));
u_t = NCNDO(p,u,dt);
x1   = NCNDO(p,x[0,:],dt);
x2   = NCNDO(p,x[1,:],dt);
y_t = NCNDO(p,y,dt);

for i in range(0,p):
    x_t[2*i,:]   = x1[i,:];
    x_t[2*i+1,:] = x2[i,:];

# Fetch Timer
Tim = Timer('Running cost estimates for case 12/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;
        
        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,1);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    

#plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Cost'],loc=10);
plt.savefig('US_DEMGD.png', dpi=300);


#%% Case 4.1: DEM with embedded history and known states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,-1),eye(2));  

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Fetch Timer
Tim = Timer('Running cost estimates for case 13/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;

        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f7 = plt.figure(7,figsize=(wfig,hfig));
plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 4.2: DEM with embedded history and known states, kernel width = 0.1 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,-1),eye(2));  

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Fetch Timer
Tim = Timer('Running cost estimates for case 14/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;

        M_t.th = th;
        xu_t[:,:] = 0;
        xp_t[:,:] = 0;
        yp_t[:,:] = 0;
        
        for k in range(1,N):
            xu_t[:,k] = dot(D,x_t[:,k-1]);
            xp_t[:,k] = M_t.f(x_t[:,k-1], u_t[:,k-1], th);
            yp_t[:,k] = M_t.h(xp_t[:,k], u_t[:,k], th);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Theoretical cost','Practical cost'],loc=10);
plt.savefig('KS_DEMGH.png', dpi=300);

#%% Case 4.3: DEM with embedded history and unknown states, kernel width = 0.01 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 1;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,-1),eye(2));  

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Fetch Timer
Tim = Timer('Running cost estimates for case 15/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;

        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,2);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
f8 = plt.figure(8,figsize=(wfig,hfig));
#plotcost(Jt, maxth,'black',2,1);
plotcost(Jp, maxth,tudcols[0],2,1);

#%% Case 4.4: DEM with embedded history and unknown states, kernel width = 0.1 s

# Case-specific parameters
maxth = 20;                     # Maximum deviation of actual param value
s = 10;

# Reset some parameters
dth = 2*maxth/(res-1);          # Parameter step-size;
Jp = zeros((3,res));            # Storage for cost
Jt = zeros((3,res));            # Storage for cost
th[:] = th_r[:];                   # reset parameters

# Get noise and simulate system (generate training data)
W = Noise(dt,T,1,1,3,pd,[],[s*dt],seed);
w_tmp = W.getNoise()[1];
w[:,:] = w_tmp[0:2,:];
v[:,:] = w_tmp[2,:];
x,y     = S.sim([0,0],u,w,v);

# Generalize Model for predictions, derivatives and history 
M_t = generalize(M,p,DTCM('GF',p,s))[0];
D = kron(eye(p,p,-1),eye(2));  

# Generalize in- and output signals;
u_t = zeros((p*M.nu,N));
x_t = zeros((p*M.nx,N));
y_t = zeros((p*M.ny,N));

# Generalized signals (history)
for i in range(0,p):
    u_t[M.nu*i:M.nu*(i+1),i:N] = u[:,0:N-i];
    x_t[M.nx*i:M.nx*(i+1),i:N] = x[:,0:N-i];
    y_t[M.ny*i:M.ny*(i+1),i:N] = y[:,0:N-i];

# Fetch Timer
Tim = Timer('Running cost estimates for case 16/16',0,3*res);

# Find cost values depending on parameter estimates
for i in range(0,3):
    
    # For current parameter, start at lowest eval value
    th[i] = th[i] - maxth-dth;
    
    for j in range(0,res):
        
       # update parameter value for current eval. values
        th[i]  = th[i] + dth;

        M_t.th = th;
        xu_t,xp_t,yu_t,yp_t = FEF(M_t,zeros(M_t.nx),u_t,y_t,p,alpha,2);
        
        Jp[i,j] = qcf(y_t,yp_t,M_t.R,xu_t,xp_t,M_t.Q);
        #Jt[i,j] = qcf(y_t,yp_t,M_t.R,x_t,xp_t,M_t.Q);
         
        Tim.looptime(i*res + j);
    
    # Reset current parameter to actual value
    th[i] = th[i] - maxth;
    
    # Normalize for plot scaling
    #Jt[i,:] = (Jt[i,:]-min(Jt[i,:]))/(max(Jt[i,:])-min(Jt[i,:]));
    Jp[i,:] = (Jp[i,:]-min(Jp[i,:]))/(max(Jp[i,:])-min(Jp[i,:]));
    
#plotcost(Jt, maxth,'black',2,2);
plotcost(Jp, maxth,tudcols[0],2,2);
plt.legend(['Cost','Practical cost'],loc=10);
plt.savefig('US_DEMGH.png', dpi=300);
