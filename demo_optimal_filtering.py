""""   
    author:   Laurens Y.D. Žnidaršič
    subject:  uncorellated noise filtering
    version:  1.0.0
    license:  Free to use
    copyright: Copyright 2020 by Laurens Y.D. Žnidaršič'
"""

from src.Noise import Noise
from numpy import zeros, eye, diag, logspace
from numpy.linalg import inv, norm
from src.myfunctions import dotn,tp
from src.visualizations import tudcols,myplot
import matplotlib.pyplot as plt
from src.filtering import dare, sgfilter
#%% Initialization

# General settings
dt = 1e-2;                              # Sampling time
T = 10;                                 # Simulation time
N = int(T/dt+1)                         # Number of samples

# System parameters
m = 0.1;                                # Mass
d = 1;                                  # Damping
k = 0.5;                                # Stiffness

# Noise sequence parameters
seed = [1234];                          # Seed for random number generator
std = [0.5,0.5,1]                       # Standard devation of noise signals

# Initialize empty storage
u  = zeros((1,N));                      # input
x  = zeros((2,N));                      # system hidden state
xp = zeros((2,N));                      # predicted hidden state
xu = zeros((2,N));                      # updated hidden state
y  = zeros((1,N));                      # system output
yp = zeros((1,N));                      # filtered output
w  = zeros((2,N));                      # system output
z  = zeros((1,N));                      # filtered output

# Get discrete-time system matrices
A = zeros((2,2)); 
A[0,1] = 1; 
A[1,0] = -k/m; 
A[1,1] = -d/m; 
A = eye(2) + dt*A;

B = zeros((2,1)); 
B[1,0] = 1/m;
B = dt*B;

C = zeros((1,2)); 
C[0,0] = 1;

D = zeros((1,1));

# Get noise signals
zeta = zeros((3,2)); zeta[:,1] = std;   # Sufficient statistics (mn, std)
Q = diag(std[0:2]); Q = dotn(tp(Q),Q);  # Covariance on state noise
R = diag([std[2]]); R = dotn(tp(R),R);  # Covariance on soutput noise
t,W = Noise(dt,T,1,1,1,zeta,[],[],seed).getNoise();
w[:,:] = W[0:1,:];                      # State noise
z[:,:] = W[2,:];                        # Output noise

# Construct input signal (step at 2 sec.)
u[0,int(2/dt+1):] = 10;

# Simulate system
y[:,0] = dotn(C,x[:,0]) + dotn(D,u[:,0]) + z[:,0];
for k in range(1,N):
    x[:,k] = dotn(A,x[:,k-1]) + dotn(B,u[:,k-1]) + w[:,k-1];
    y[:,k]   = dotn(C,x[:,k]) + dotn(D,u[:,k]) + z[:,k];

# General figure settings
plt.close('all');
hfig = 6;
wfig = 18;
font = {'size'   : 14}
plt.rc('font', **font);
plt.rc('axes', titleweight='bold');
plt.rc('xtick', labelsize=10);
plt.rc('ytick', labelsize=10);
xbet = 0.03; 
ybet = 0.07
lef = 0.05; 
bot = 0.08; 
top = 0.01;

#%% Case 1: Compare Kalman Covariance estimates with Q and R

normP = zeros(N);                      # filtered output
normS = zeros(N);                      # filtered output
normK = zeros(N);                      # filtered output
P = zeros((2,2));
S = zeros((1,1));
K = zeros((2,1));

for k in range(0,N):
    normP[k] = norm(P,2);
    normS[k] = norm(S,2);
    S = dotn(C,P,tp(C)) + R;
    P = dotn(A,P,tp(A)) + Q;
    K = dotn(P,tp(C),inv(S));
    P = dotn(eye(2)-dotn(K,C),P);
  
# Case-specific figure + settings
plt.figure(1,figsize=(wfig,hfig));
w = (1-lef-2*xbet)/2;
ax1 = plt.axes((lef,bot,w,1-top-bot));
ax2 = plt.axes((lef+xbet+w,bot,w,1-top-bot));

myplot(ax1,t[[0,-1]],[norm(Q,2),norm(Q,2)],'black');
myplot(ax1,t,normP,tudcols[0],'t (s)','$||P||$',\
       'Converging behaviour of Kalman covariance estimate','linlin');
plt.legend(['||Q||','||P||']);
myplot(ax2,t[[0,-1]],[norm(R,2),norm(R,2)],'black');
myplot(ax2,t,normS,tudcols[0],'t (s)','$||S||$',\
       'Converging behaviour of Kalman covariance estimate','linlin');
plt.legend(['||R||','||S||']);

#%% Case 2: try out different values for K and store error
Nk = 51;
alpha = logspace(-0.5,0.5,Nk)
ex = zeros(Nk);      # state estimation error

K,P,S = dare(A,C,Q,R);

# Simulate filter for different values of K
for i in range(0,Nk):
    xp,xu,yp,yu = sgfilter(A,B,C,D,alpha[i]*K,[0,0],u,y);
    for k in range(1,N):
        ex[i]  += ((x[0,k]-xu[0,k])**2 + (x[1,k]-xu[1,k])**2)/N;
        
# Case-specific figure + settings
plt.figure(2,figsize=(wfig,hfig));
ax1 = plt.axes((lef,bot,1-lef-xbet,1-top-bot));
myplot(ax1,alpha,ex,tudcols[0],'','MSE',\
       'MSE on state estimate','loglin',\
       [[0.5,1,2],["$0.5K^*$","$K^*$",\
        "$2K^*$"]],[]);
       
# Repeat for sub-optimal gain constructed from the noise covariances
#ex = zeros(Nk);      # state estimation error
#K = dotn(Q,tp(C),inv(R)); 
#for i in range(0,Nk):
#    for k in range(1,N):
#        xp[:,k] = dotn(A,xu[:,k-1]) + dotn(B,u[:,k-1]);
#        yp[:,k] = dotn(C,xp[:,k]) + dotn(D,u[:,k]);
#        xu[:,k] = xp[:,k] + dotn(alpha[i]*K,y[:,k] - yp[:,k]);
#        ex[i]  += ((x[0,k]-xu[0,k])**2 + (x[1,k]-xu[1,k])**2)/N;
#      
#myplot(ax1,alpha,ex,tudcols[5],'\u03B1','MSE',\
#       'MSE on state estimate','loglin',\
#       [[0.5,1,2],["$0.5K^*$","$K^*$",\
#        "$2K^*$"]],[]);
#
#plt.legend(["$K^* = PC^TS^{-1}$", "$K^* = QC^TR^{-1}$"]);

plt.savefig('em.png', dpi=300);

