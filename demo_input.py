""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  Persistently exciting input signal used in simulations
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from numpy import zeros, linspace
from src.signals import getSignal
from src.visualizations import tudcols
import matplotlib.pyplot as plt

# General Settings
dt  = 1e-2;                     # Time-step
T   = 10.0;                     # Simulation period
N = int(T/dt+1);
t = linspace(0,T,N);

# Get persistently exciting input signal (sum of sinoids)
u = zeros((1,N));
u[0,:] = getSignal("LOGSINSUM",t,5,0.1,3,5,1);

# General figure settings
plt.close('all');
hfig = 3;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);

#%%

f1 = plt.figure(1,figsize=(wfig,hfig));
ax = plt.axes([0.03,0.12,0.96,0.87]);
plt.plot(t,u[0,:],color=tudcols[0]);
plt.xlim([0,t[-1]]);
ymin = min(u[0,:]);
h = max(u[0,:]) - ymin;
plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
plt.title('Input: u',pad=-16);
ax.tick_params(which='both',direction='in'); 
ax.xaxis.set_label_coords(0.5, -0.05);
plt.xlabel('t (s)');
    
plt.savefig('input.png', dpi=300);