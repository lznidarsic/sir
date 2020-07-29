""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  sysid with uncorellated noise using expectation maximization
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""
from numpy import zeros
from src.Noise import Noise
import matplotlib.pyplot as plt

# General Settings
dt  = 1e-2;                     # Time-step
T   = 10.0;                     # Simulation period

# Noise parameters
zeta = [1];
seed = [1234];

# Set vectors and dimensions
N = int(1+T/dt)

# Noise category parameters
titles = ['White Noise', 'Gaussian Noise', 'Pink Noise', 'Brownian Noise']
pf = [1,1,2,1];
pa = [1,3,1,2];
s = 0.01;

# Construct Noise
pd = zeros((1,2));
pd[:,1] = zeta;
Q = diag(pd[0:2,1]); Q = Q**2;

# General figure settings
plt.close('all');
hfig = 3;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);
lef = 0.03;
bot = 0.12;
bet = 0.01;
hf = (1-bot-3*bet) /3
wf = 1-lef-bet;
f1 = plt.figure(1,figsize=(wfig,hfig));

#%% Case 1: kernel width = 1
for i in range(0,4):
    W = Noise(dt,T,1,1,3,pd,[],[1*dt],seed);
    t,w = W.getNoise();


ax = plt.axes([lef,bot,wf,hf]);
plt.plot(t,w1[0,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(w1[0,:]);
h = max(w1[0,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('Output noise: $z$',pad=-16);
ax.tick_params(which='both',direction='in'); 
ax.xaxis.set_label_coords(0.5, -0.05);
plt.xlabel('t (s)');

ax = plt.axes([lef,bot+hf+bet,wf,hf]);
plt.plot(t,w1[1,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(w1[1,:]);
h = max(w1[1,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.xticks([],[]);
plt.title('State noise: $w_2$',pad=-16);
ax.tick_params(which='both',direction='in'); 

ax = plt.axes([lef,bot+2*(hf+bet),wf,hf]);
plt.plot(t,v1[0,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(v1[0,:]);
h = max(v1[0,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('State noise: $w_1$',pad=-16);
plt.xticks([],[]);
ax.tick_params(which='both',direction='in'); 

plt.savefig('noise1.png', dpi=300);

#%% Case 2: kernel width = 10
f2 = plt.figure(2,figsize=(wfig,hfig));

W10 = Noise(dt,T,1,1,3,pd,[],[10*dt],seed);
t,w_tmp10 = W10.getNoise();
w10[:,:] = w_tmp10[0:2,:];
v10[:,:] = w_tmp10[2,:];

ax = plt.axes([lef,bot,wf,hf]);
plt.plot(t,w10[0,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(w10[0,:]);
h = max(w10[0,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('Output noise: $z$',pad=-16);
ax.tick_params(which='both',direction='in'); 
ax.xaxis.set_label_coords(0.5, -0.05);
plt.xlabel('t (s)');

ax = plt.axes([lef,bot+hf+bet,wf,hf]);
plt.plot(t,w10[1,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(w10[1,:]);
h = max(w10[1,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('State noise: $w_2$',pad=-16);
plt.xticks([],[]);
ax.tick_params(which='both',direction='in'); 

ax = plt.axes([lef,bot+2*(hf+bet),wf,hf]);
plt.plot(t,v10[0,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(v10[0,:]);
h = max(v10[0,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('State noise: $w_1$',pad=-16);
plt.xticks([],[]);
ax.tick_params(which='both',direction='in'); 

plt.savefig('noise10.png', dpi=300);