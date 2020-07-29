""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  corellated noise demo using Noise generator
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from src.Noise import Noise
from src.visualizations import showNoiseData;
from numpy import zeros;
import matplotlib.pyplot as plt;

# Noise sequence parameters
seed = [1234];                              # Seed for random number generator
zeta = zeros((1,2)); zeta[:,:] = [0,1];     # Sufficient statistics (mn, std)
dt = 1e-3;                                  # Sampling time
T = 10;                                     # Simulation time
sigma = 0.01;                                # Kernel width (case 6 and 7)


# Figure settings
plt.close('all');
hfig = 6;
wfig = 18;

# Case 1: White noise
plt.figure(1,figsize=(wfig,hfig));
White = Noise(dt,T,1,1,1,zeta,[],[],seed);
showNoiseData(White);
plt.savefig('whitenoise.png', dpi=300);

# Case 2: Pink noise
plt.figure(2,figsize=(wfig,hfig));
Pink   = Noise(dt,T,1,2,1,zeta,[],[],seed);
showNoiseData(Pink);
plt.savefig('pinknoise.png', dpi=300);

# Case 3: Red noise
plt.figure(3,figsize=(wfig,hfig));
Red = Noise(dt,T,1,1,2,zeta,[],[],seed);
showNoiseData(Red,corwin=20);
plt.savefig('rednoise.png', dpi=300);

# Case 4: Blue noise
plt.figure(4,figsize=(wfig,hfig));
Blue = Noise(dt,T,1,4,1,zeta,[],[],seed);
showNoiseData(Blue);
plt.savefig('bluenoise.png', dpi=300);

# Case 5: White Violet noise
plt.figure(5,figsize=(wfig,hfig));
Violet = Noise(dt,T,1,5,1,zeta,[],[],seed);
showNoiseData(Violet);
plt.savefig('violetnoise.png', dpi=300);

# Case 6: Gaussian noise
plt.figure(6,figsize=(wfig,hfig));
Gaussian = Noise(dt,T,1,1,3,zeta,[],[sigma],seed);
showNoiseData(Gaussian);
plt.savefig('gaussiannoise.png', dpi=300);

# Case 7: Block noise
plt.figure(7,figsize=(wfig,hfig));
Block = Noise(dt,T,1,1,4,zeta,[],[sigma],seed);
showNoiseData(Block);
plt.savefig('blocknoise.png', dpi=300);
