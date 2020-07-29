from numpy  import zeros, sin, cos
from src.Model import Model

dt = 1e-2;

#%% Spring-mass-damper system
def f_smd(x,u,th):
    xp = zeros(2);
    xp[0] =           x[0] +           dt*x[1];
    xp[1] = -dt*th[0]*x[0] + (1-dt*th[1])*x[1] + dt*th[2]*u[0];
    return xp;

def A_smd(x,u,th):    
    A = zeros((2,2));
    A[0,0] = 1;          A[0,1] = dt;
    A[1,0] = -dt*th[0];  A[1,1] = 1-dt*th[1];
    return A;

def B_smd(x,u,th):    
    B = zeros((2,1));
    B[1,0] = dt*th[2];
    return B;

def F_smd(x,u,th):
    F = zeros((2,3));
    F[1,0] = -dt*x[0];
    F[1,1] = -dt*x[1];
    F[1,2] =  dt*u[0];
    return F;

def h_smd(x,u,th):
    return x[0];

def C_smd(x,u,th): 
    C = zeros((1,2));
    C[0,0] = 1;
    return C;

def D_smd(x,u,th): 
    D = zeros((1,1));
    return D;

def H_smd(x,u,th):
    H = zeros((1,3));
    return H;

#%% mass-nonlinear spring- damper system

def f_nlsmd(x,u,p):    
    xp = zeros(2);
    xp[0] = x[0] + dt*(x[1]);
    xp[1] = x[1] + dt*(-p[0]*x[0]**3- p[1]*x[1] + p[2]*u[0]);
    return xp;

def dfx_nlsmd(x,u,p):    
    F = zeros((2,2));
    F[0,0] = 1;
    F[0,1] = dt;
    F[1,0] = -3*dt*p[0]*x[0]**2;
    F[1,1] = -  dt*p[1];
    return F;

def dfp_nlsmd(x,u,p):
    np = len(p);
    Fp = zeros((2,np));
    Fp[1,0] = -dt*x[0]**3;
    Fp[1,1] = -dt*x[1];
    Fp[1,2] =  dt*u[0];
    return Fp;

def h_nlsmd(x,u,p):
    return x[0];

def dhx_nlsmd(x,u,p): 
    H = zeros((1,2));
    H[0,0] = 1;
    return H;

def dhp_nlsmd(x,u,p):
    np = len(p);
    Hp = zeros((1,np));
    return Hp;

#%% Pendulum (with gravity and damping, no spring)

def f_pen(x,u,p):    
    xp = zeros(2);
    xp[0] = x[0] + dt*(x[1]);
    xp[1] = x[1] + dt*(-p[0]*sin(x[0])- p[1]*x[1] + p[2]*u[0]);
    return xp;

def dfx_pen(x,u,p):
    F = zeros((2,2));
    F[0,0] = 1;
    F[0,1] = dt;
    F[1,0] = - dt*p[0]*cos(x[0]);
    F[1,1] = - dt*p[1];
    return F;

def dfp_pen(x,u,p):
    np = len(p);
    Fp = zeros((2,np));
    Fp[1,0] = -dt*x[0]**3;
    Fp[1,1] = -dt*x[1];
    Fp[1,2] =  dt*u[0];
    return Fp;

def h_pen(x,u,p):
    return x[0];

def dhx_pen(x,u,p): 
    H = zeros((1,2));
    H[0,0] = 1;
    return H;

def dhp_pen(x,u,p):
    np = len(p);
    Hp = zeros((1,np));
    return Hp;

#%% Retrieve a system
def getModel(ident):
    if ident == "SMD":
        print("Retrieved: spring-mass-damper system" +\
              "\n   p[0]   k/m" +\
              "\n   p[1]   d/m"+\
              "\n   p[2]   1/m\n");
        M = Model("Spring-Mass-Damper System",'ltiss',dt,[2,1,1,3],f_smd, A_smd, B_smd,\
                  F_smd, h_smd, C_smd, D_smd, H_smd); 
        return M;
    elif ident == "NLSMD":
        print("Retrieved: non-linear spring-mass-damper system" +\
              "\n   p[0]   k/m" +\
              "\n   p[1]   d/m"+\
              "\n   p[2]   1/m\n");
        M = Model("Non-linear Spring-Mass-Damper System",dt,[2,1,1,3],f_nlsmd, \
                  dfx_nlsmd, dfp_nlsmd, h_nlsmd, dhx_nlsmd, dhp_nlsmd); 
        return M;
    elif ident == "PEN":
        print("Retrieved: pendulum system" +\
              "\n   p[0]   g/l" +\
              "\n   p[1]   d/(ml^2)"+\
              "\n   p[2]   1/(ml^2)\n");
        M = Model("Pendulum without spring",dt,[2,1,1,3],f_pen, \
                  dfx_pen, dfp_pen, h_pen, dhx_pen, dhp_pen);
        return M;
    else:
        print("Unknown system identifier. Options are: "+\
              "\n   SMD     spring-mass-damper system" +\
              "\n   NLSMD   Non-linear spring-mass-damper system" +\
              "\n   PEN     Pendulum\n" );
