""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  derivative observers, numerical v.s. stable
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from numpy import linspace, sin,pi, zeros
from src.parameterestimation import SDO,NDO
from src.visualizations import tudcols, plotGS
import matplotlib.pyplot as plt

# Parameters
w  = 2;
T  = 10.0
dt = 1e-2;
p  = 4;
N = int(T/dt)+1;
t = linspace(0,T,N);

# Signal
u = zeros((1,N));
u[0,:] = sin(w*t);
u_t = zeros((p,N));
for i in range(0,p):
    u_t[i,:] = (w**i)*sin(w*t**2 + i*pi/2);

# General figure settings
plt.close('all');
hfig = 5;
wfig = 18;
font = {'size'   : 13}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);

#%% Case 1: Numerical differentiator, dt = 1e-2

# Estimate generalized signal
u_tn  = NDO(p,u,dt)

# Plot
f1 = plt.figure(1,figsize=(wfig,hfig));
plotGS(t,u_t,'black',True);
plotGS(t,u_tn,tudcols[0],False);
plt.legend(['Truth', 'Numerical filter']);
plt.savefig('numerfilt01', dpi=300);

#%% Case 2: Stable filter, dt = 1e-2

# Estimate generalized signal

u_ts200  = SDO(p,u,zeros(p),dt,200)
u_ts50  = SDO(p,u,zeros(p),dt,50)
u_ts10  = SDO(p,u,zeros(p),dt,10)

# Plot
f2 = plt.figure(2,figsize=(wfig,hfig));
plotGS(t,u_t,'black',True);
plotGS(t,u_ts200,tudcols[0],False);
plotGS(t,u_ts50,tudcols[0],False,'--');
plotGS(t,u_ts10,tudcols[0],False,':');
plt.legend(['Truth','Stable filter, $\lambda$ = 50 rad/s','Stable filter, $\lambda$ = 10 rad/s']);
plt.savefig('stablefilt01.png', dpi=300);

#%% Intermezzo: reset conditions for case 3 and 4

# Parameters
dt = 1e-3;
N = int(T/dt)+1;
t = linspace(0,T,N);

# Signal
u = zeros((1,N));
u[0,:] = sin(w*t);
u_t = zeros((p,N));
for i in range(0,p):
    u_t[i,:] = (w**i)*sin(w*t + i*pi/2);

#%% Case 3: Numerical differentiator, dt = 1e-3

# Estimate generalized signal
u_tn  = NDO(p,u,dt)

# Plot
f3 = plt.figure(3,figsize=(wfig,hfig));
plotGS(t,u_t,'black',True);
plotGS(t,u_tn,tudcols[0],False);
plt.legend(['Truth', 'Numerical filter']);
plt.savefig('numerfilt001', dpi=300);

#%% Case 4: Stable filter, dt = 1e-3

# Estimate generalized signal

u_ts200  = SDO(p,u,zeros(p),dt,200)
u_ts50  = SDO(p,u,zeros(p),dt,50)
u_ts10  = SDO(p,u,zeros(p),dt,10)

# Plot
f4 = plt.figure(4,figsize=(wfig,hfig));
plotGS(t,u_t,'black',True);
plotGS(t,u_ts200,tudcols[0],False);
plotGS(t,u_ts50,tudcols[0],False,'--');
plotGS(t,u_ts10,tudcols[0],False,':');
plt.legend(['Truth','Stable filter, $\lambda$ = 200 rad/s','Stable filter, $\lambda$ = 10 rad/s']);
plt.savefig('stablefilt001.png', dpi=300);
