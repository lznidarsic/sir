from numpy import zeros, dot
from src.myfunctions import isempty

class Model:

    __author__    = 'Laurens Y.D. Žnidaršič';
    __title__     = 'Model class';
    __version__   = '1.0.0';
    __license__   = 'Free to use';
    __copyright__ = 'Copyright 2020 by Laurens Y.D. Žnidaršič';
    
    __slots__ = ['name','typ','f','A','B','F','h','C','D','H','th','dt','Q','R','Qth','Qx',\
                 'nx','nu','ny','nth'];
    """
    Initializer / Instance Attributes
    name   Short description of the system
    f      transition model structure equations f(x,u,th)
    h      measurement model structure equations h(x,u,th)
    A      df/dx
    B      df/du
    F      df/dth
    C      dh/dx
    D      dh/du
    H      dh/dth
    dt     time-step. continuous time not implemented (yet)
    pf     transition model parameter vector 
    ph     measurement model parameter vector 
    """
    def __init__(self,name="Nameless System",typ='ltiss',dt=1e-3,
                 dim=[0,0,0,0], f=[],A=[],B=[],F=[],h=[],C=[],D=[],H=[],\
                 th=[],Q=[],R=[],Qx=[],Qth=[]):
        
        self.nx = dim[0];
        self.nu = dim[1];
        self.ny = dim[2];
        self.nth = dim[3];
        
        self.name = name;
        self.typ = typ;
        self.dt = dt;
        
        self.f = f;
        self.h = h;
        self.A = A;
        self.B = B;
        self.C = C;
        self.D = D;
        self.F = F;
        self.H = H;
        
        if isempty(th):
           self. th = zeros(self.nth);
        else:
            self.th = th;

        self.setQ(Q);
        self.setR(R);
        
        return;
        
    def __str__(self):
        dt = self.dt*1e3;
        dt = " % 2.3f" % dt;
        ret = self.__repr__() + "\n";
        if self.f != [] and self.h != []:   
            ret = ret +  "f and h specified:    yes\n"
        else:
            ret = ret +  "f and h specified:    no\n"
        if self.pf != [] and self.ph != []:   
            ret = ret +  "parameters specified: yes\n"
        else:
            ret = ret +  "parameters specified: no\n"
        ret = ret + "Model order:          " + str(self.nx) + "\n";
        ret = ret + "Inthut dim.:           " + str(self.nu) + "\n";
        ret = ret + "Output dim:           " + str(self.ny) + "\n";
        ret = ret + "Time step:          " + dt           + " ms\n";

            
        return ret
        
    def __repr__(self):
        return self.name + " instance of Model Class";

    def setP(self,th):
        self.th = th;
        return;
        
    def setQ(self,Q):
        self.Q = zeros((self.nx,self.nx));
        if len(Q) != 0:
            self.Q[:,:] = Q; 
        return;
        
    def setR(self,R):
        self.R = zeros((self.ny,self.ny));
        if len(R) != 0:
            self.R[:,:] = R; 
        return;
        
        
    def getFunctions(self):
        return  self.f,self.dfx,self.dfp,self.h,self.dhx,self.dhp;
      
        
    def __dimcheck(self,x,nx,N,name):
        if N == 1:
            dflt = zeros(nx);
        else:
            dflt = zeros((nx,N));
        if len(x) == 0:
            x = dflt;
        elif len(x) != nx:
            x = dflt;
            print("WARNING: wrong "+name+" dimensionaliy. "+\
                  "Setting default")  
        else:
            pass; # all is fine
        return x;
        
    # Model simulator
    #
    # Inthuts
    # t      time sequence
    # u      inthut sequence
    # nmax   maximum order estimation def=10
    #
    # Outputs
    # nx    state dimensionality (model order)
    # nu    inthut dimensionality
    # ny    output dimensionality 
    def sim(self,x0,u,w=[],v=[],show=0):
        N = len(u[0,:]);
        
        x = zeros((self.nx,N)); x[:,0] = x0;
        y = zeros((self.ny,N)); y[:,0] = self.h(x0,u[:,0],self.th)+ v[:,0];
        for k in range(1,N):
            x[:,k] = self.f(x[:,k-1],u[:,k-1],self.th) + w[:,k-1];
            y[:,k]   = self.h(x[:,k],u[:,k],self.th)   + v[:,k];    
        return x,y;
    
    # Model simulator
    #
    # Inthuts
    # t      time sequence
    # u      inthut sequence
    # nmax   maximum order estimation def=10
    #
    # Outputs
    # nx    state dimensionality (model order)
    # nu    inthut dimensionality
    # ny    output dimensionality 
    def filt(self,K,x0,u,y):
        N = len(u[0,:]);      
        xp = zeros((self.nx,N)); xp[:,0] = x0;
        xu = zeros((self.nx,N)); xu[:,0] = x0;
        yp = zeros((self.ny,N)); yp[:,0] = self.h(x0,u[:,0],self.th);
        yu = zeros((self.ny,N)); yu[:,0] = self.h(x0,u[:,0],self.th);
        
        for k in range(1,N):
            xp[:,k] = self.f(xu[:,k-1],u[:,k-1],self.th);
            yp[:,k] = self.h(xp[:,k],  u[:,k],  self.th);
            xu[:,k] = xp[:,k] + dot(K,y[:,k]-yp[:,k]);
            yu[:,k] = self.h(xu[:,k],  u[:,k],  self.th);
        return xp,xu,yp,yu;
