from src.Timer import Timer
from numpy import zeros,dot, eye, transpose, exp
from numpy import kron,log
from numpy.linalg import inv, norm
from scipy.linalg import expm
from src.Model import Model
from src.myfunctions import dotn, tp, isempty, fct, powdiag, pascal,NG, dare,isint
from src.pedata import pedata;


#%% Parameter estimation

def EM(M,x0,u,y,Q,R,alpha=1,maxit=1000,numerical=True,tol=1e-6,conit=100,gembed=1):
    
    """
    Expectation Maximization
    INPUTS
      M            Data-structure of Model class
      x0           Initial hidden state - 1-dimensional array_like (list works)
      u            Input sequence nu x N numpy_array 
      y            Output sequence ny x N numpy_array 
      Q            State-noise covariance - nx x nx numpy_array 
      R            Output-noise covariance - ny x ny numpy_array 
      alpha        Updating gain - scalar float, int - (opt. def=1)
      maxit        Max. number of iterations - scalar int - (opt. def=1000)
      numerical    Use numerical gradients - boolean - (opt. def=True)
      tol          Error tolerance for stopping cond. - float - (opt. def=True)
      conit        Number of converence iterations. - fint - (opt. def=100)
      gembed       Gradient-embedding order (if algebraical gradient )
                                                  - scalar int - (opt. def=1)
    OUTPUTS
      dat          Data-structure of pedata class containing
    """
    
    # Allocate memory
    J  = zeros((    1,       maxit));
    K  = zeros(( M.nx, M.ny, maxit));
    P  = zeros(( M.nx, M.nx, maxit));
    S  = zeros(( M.ny, M.ny, maxit));
    th = zeros((M.nth,       maxit));
    th[:,0] = M.th;
    
    # Set-up loop
    i = 0;
    j = 0;
    T = Timer('Expectation Maximization',i,maxit-1,maxit-1);
    
    # Run loop
    while i < maxit-1 and j < conit:
        
        # Update cost, covariance and parameter estimates
        if numerical:
            J[:,i],K[:,:,i],P[:,:,i],S[:,:,i],th[:,i+1] \
                    = NEMstep(M,x0,u,y,Q,R,alpha)[0:5];
        else:
            J[:,i],K[:,:,i],P[:,:,i],S[:,:,i],th[:,i+1] \
                    = AEMstep(M,x0,u,y,Q,R,alpha,gembed)[0:5];
        
        M.th = th[:,i+1];
    
        # Stopping criterion: J < <tol> in last <conit> iterations
        if j == 0:
            if abs(J[:,i]-J[:,i-1]) < tol:
                j = 1;
        else:
            if abs(J[:,i]-J[:,i-1]) < tol:
                j +=1;
            else:
                j = 0;
        i +=1;
        T.looptime(i);
    
    # Get final hidden state and output estimations
    xp,xu,yp,yu = M.filt(K[:,:,i],x0,u,y);
    
    # If converged before maxit, then fill remainer of matrices with last value
    for j in range(i-1,maxit):
        J[:,j]   = J[:,i-1]
        K[:,:,j] = K[:,:,i-1]
        P[:,:,j] = P[:,:,i-1]
        S[:,:,j] = S[:,:,i-1]
        th[:,j]  = th[:,i]
    
    dat = pedata('EM',M,J,K,P,S,th,xp,xu,yp,yu);
    return dat;
    

def NEMstep(M,x0,u,y,Q,R,alpha):
    
    """
    # 1-step Numerical-gradient Expectation-Maximization 
    INPUTS
      M            Data-structure of Model class
      x0           Initial hidden state - 1-dimensional array_like (list works)
      u            Input sequence nu x N numpy_array 
      y            Output sequence ny x N numpy_array 
      Q            State-noise covariance - nx x nx numpy_array 
      R            Output-noise covariance - ny x ny numpy_array 
      alpha        Updating gain - scalar float, int - (opt. def=1)
    OUTPUTS
      J            Cost (log-likelihood) - scalar float
      K            Kalman gain - nx x ny numpy_array 
      P            State-error covariance estimate - nx x nx numpy_array 
      S            Output-error covariance estimate - nx x nx numpy_array  
      th           Updated parameter estimate - 1-dimensional numpy_array  
      xp           Predicted hidden state estimates - nx x N numpy_array 
      xu           Updated hidden state estimates - nx x N numpy_array 
      xp           Predicted output estimates - ny x N numpy_array 
      xu           Updated output state estimates - ny x N numpy_array  
    """
    
    N =  len(u[0,:]);

    # Define cost as a function of theta
    def cost(th):
        
        K,P,S,xp,xu,yp,yu = KF(M,x0,u,y,th);
        
        # Calculate cost
        J = ll(y,yp,R,xu,xp,Q);
        
        return J,K,P,S,xp,xu,yp,yu;

    # Get parameter gradient
    dth = NG(cost,M.th)/N;
    
    # Update parameters
    th = M.th + alpha*dth;

    # Calculate all other variables
    J,K,P,S,xp,xu,yp,yu = cost(M.th);

    return J,K,P,S,th,xp,xu,yp,yu;  

# 1-step algebraical gradient Expectation-Maximization 
def OEMstep(M,u,xu,xp,y,yp,th,iQ,iR,P,alpha=1):
    
    """
    # 1-step Algebraical-gradient Expectation-Maximization 
    INPUTS
      M            Data-structure of Model class
      x0           Initial hidden state - 1-dimensional array_like (list works)
      u            Input sequence nu x N numpy_array 
      y            Output sequence ny x N numpy_array 
      Q            State-noise covariance - nx x nx numpy_array 
      R            Output-noise covariance - ny x ny numpy_array 
      alpha        Updating gain - scalar float, int - (opt. def=1)
      gembed       Gradient-embedding order - scalar int - (opt. def=1)
    OUTPUTS
      J            Cost (log-likelihood) - scalar float
      K            Kalman gain - nx x ny numpy_array 
      P            State-error covariance estimate - nx x nx numpy_array 
      S            Output-error covariance estimate - nx x nx numpy_array  
      th           Updated parameter estimate - 1-dimensional numpy_array  
      xp           Predicted hidden state estimates - nx x N numpy_array 
      xu           Updated hidden state estimates - nx x N numpy_array 
      xp           Predicted output estimates - ny x N numpy_array 
      x
      """ 
    
    N =  len(u[0,:]);

    # Fetch state estimates based on current parameter estimates
    K,P,S,xp[:,-1],xu[:,-1],yp[:,-1],yu = KFstep(M,xu[:,-2],u[:,-2:],y[:,-1],P);
    
    # Initialize gradients
    dxdth = zeros((M.nx,M.nth));
    dydth = zeros((M.ny,M.nth));
    
    # Pre-calculate all matrices (trade storage for efficiency)
    F = zeros((M.nx,M.nth,N));
    H = zeros((M.ny,M.nth,N));
    A = zeros((M.nx,M.nx,N));
    B = zeros((M.nx,M.nu,N));
    C = zeros((M.ny,M.nx,N));
    D = zeros((M.ny,M.nu,N));
    
    for k in range(0,N):
        F[:,:,k] = M.F(xu[:,k],u[:,k],th[:,k]);
        H[:,:,k] = M.H(xu[:,k],u[:,k],th[:,k]);
        A[:,:,k] = M.A(xu[:,k],u[:,k],th[:,k]);
        B[:,:,k] = M.B(xu[:,k],u[:,k],th[:,k]);
        C[:,:,k] = M.C(xu[:,k],u[:,k],th[:,k]);
        D[:,:,k] = M.D(xu[:,k],u[:,k],th[:,k]);
        
    # Calculate FE-to-parameter gradient
    for k in range(0,N-1):
        dxdth[:,:] = F[:,:,k] + dotn(A[:,:,k],dxdth)
    dydth[:,:]     = - H[:,:,-1] - dotn(C[:,:,-1],dxdth);
    dxdth[:,:]     = - F[:,:,-1] - dotn(A[:,:,-1],dxdth);

    # Calculate error
    ex = xu[:,-1] - xp[:,-1];
    ey = y[:,-1]  - yu;
    
    # update gradient
    dth = dotn(tp(dxdth),inv(P),ex) + dotn(tp(dydth),inv(S),ey);

    # Update parameters
    th[:,-1] = M.th - alpha*dth;

    # Calculate cost
    J = ll(y,yp,iR,xu,xp,iQ);

    return J,K,P,S,th[:,-1],xp[:,-1],xu[:,-1],yp[:,-1],yu;  

# 1-step algebraical gradient Expectation-Maximization 
def AEMstep(M,x0,u,y,Q,R,alpha=1,gembed=1):
    
    """
    # 1-step Algebraical-gradient Expectation-Maximization 
    INPUTS
      M            Data-structure of Model class
      x0           Initial hidden state - 1-dimensional array_like (list works)
      u            Input sequence nu x N numpy_array 
      y            Output sequence ny x N numpy_array 
      Q            State-noise covariance - nx x nx numpy_array 
      R            Output-noise covariance - ny x ny numpy_array 
      alpha        Updating gain - scalar float, int - (opt. def=1)
      gembed       Gradient-embedding order - scalar int - (opt. def=1)
    OUTPUTS
      J            Cost (log-likelihood) - scalar float
      K            Kalman gain - nx x ny numpy_array 
      P            State-error covariance estimate - nx x nx numpy_array 
      S            Output-error covariance estimate - nx x nx numpy_array  
      th           Updated parameter estimate - 1-dimensional numpy_array  
      xp           Predicted hidden state estimates - nx x N numpy_array 
      xu           Updated hidden state estimates - nx x N numpy_array 
      xp           Predicted output estimates - ny x N numpy_array 
      x
      """ 
    
    N =  len(u[0,:]);
    
    xu = zeros((M.nx,N));
    xp = zeros((M.nx,N));
    yu = zeros((M.ny,N));
    yp = zeros((M.ny,N));
    #th = zeros((M.nth,N));
    th = M.th;
    
#    P = zeros((2,2));
#    
#    for k in range(1,N):
#        
#        
#        Nk = min(k,gembed);
#        J,K,P,S,th[:,k],xp[:,k],xu[:,k],yp[:,k],yu[:,k] = OEMstep(M,u[:,k-Nk:k+1],\
#                                          xu[:,k-Nk:k+1],xp[:,k-Nk:k+1],y[:,k-Nk:k+1],\
#                                         yp[:,k-Nk:k+1],th[:,k-Nk:k+1],iQ,iR,P,alpha)
#        
#        M.th = th[:,k];
#        
#    J = ll(y,yp,iR,xu,xp,iQ);
#    
#    return J,K,P,S,th[:,-1],xp,xu,yp,yu;
#        
    # Fetch state estimates based on current parameter estimates
    K,P,S,xp,xu,yp,yu = KF(M,x0,u,y,th);
    
    iQ = inv(P);
    iR = inv(S);
    
    # Initialize gradients
    dth   = zeros(M.nth);
    dxdth = zeros((M.nx,M.nth));
    dydth = zeros((M.ny,M.nth));
    
    # Pre-calculate all matrices (trade storage for efficiency)
    F = zeros((M.nx,M.nth,N));
    H = zeros((M.ny,M.nth,N));
    A = zeros((M.nx,M.nx,N));
    B = zeros((M.nx,M.nu,N));
    C = zeros((M.ny,M.nx,N));
    D = zeros((M.ny,M.nu,N));
    
    for k in range(0,N):
        F[:,:,k] = M.F(xu[:,k],u[:,k],th);
        H[:,:,k] = M.H(xu[:,k],u[:,k],th);
        A[:,:,k] = M.A(xu[:,k],u[:,k],th);
        B[:,:,k] = M.B(xu[:,k],u[:,k],th);
        C[:,:,k] = M.C(xu[:,k],u[:,k],th);
        D[:,:,k] = M.D(xu[:,k],u[:,k],th);
        
    # Calculate FE-to-parameter gradient
    for k in range(0,N-1):
        
        # Errors
        ex = xu[:,k+1] - xp[:,k+1];
        ey =  y[:,k  ] - yu[:,k  ];
        
        # State-to-parameter gradients
        dxdth[:,:] = 0;

        kmin = max(0,k-gembed);
        for i in range(kmin,k):
            dxdth[:,:] =   F[:,:,i] + dotn(A[:,:,i],dxdth);
        dydth[:,:]     = - H[:,:,k] - dotn(C[:,:,k],dxdth);
        dxdth[:,:]     = - F[:,:,k] - dotn(A[:,:,k],dxdth);

        # update gradient
        dth[:] += dotn(tp(dxdth),iQ,ex) + dotn(tp(dydth),iR,ey);

    # Update parameters
    th[:] = th - alpha*dth;

    # Calculate cost
    J = ll(y,yp,iR,xu,xp,iQ);

    return J,K,P,S,th,xp,xu,yp,yu;  

# Dynamic Expectation Maximization
def DEM(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed=0,gembed=1,maxit=1000,numerical=True,mfts=False,tol=1e-6,conit=100):
    
    # Allocate memory
    J  = zeros((    1,       maxit));
    th = zeros((M.nth,       maxit));
    th[:,0] = M.th;
    
    # Set-up loop
    i = 0;
    j = 0;
    T = Timer('Dynamic Expectation Maximization',i,maxit-1,maxit-1);
    
    # Run loop
    while i < maxit-1 and j < conit:
        
        # Update cost, covariance and parameter estimates
        if numerical:
            J[:,i],th[:,i+1] = NDEMstep(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed,mfts)[0:2];
        else:
            J[:,i],th[:,i+1] = ADEMstep(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed,gembed,mfts)[0:2]
        M.th = th[:,i+1];
    
        # Stopping criterion: J < <tol> in last <conit> iterations
        if j == 0:
            if abs(J[:,i]-J[:,i-1]) < tol:
                j = 1;
        else:
            if abs(J[:,i]-J[:,i-1]) < tol:
                j +=1;
            else:
                j = 0;
        i +=1;
        T.looptime(i);
    
    # Get final hidden state and output estimations
    xp,xu,yp,yu = FEF(M,x0,u,y,p,alpha_x,embed,mfts);
            
    # If converged before maxit, then fill remainer of matrices with last value
    for j in range(i-1,maxit):
        J[:,j]   = J[:,i-1];
        th[:,j]  = th[:,i];
    
    dat = pedata('DEM',M,J,[],[],[],th,xp,xu,yp,yu);
    return dat;

# 1-step numerical gradient Expectation-Maximization 
def NDEMstep(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed=0,mfts=False):
    
    N =  len(u[0,:]);

    # Define cost as a function of theta
    def cost(th):

        xp,xu,yp,yu = FEF(M,x0,u,y,p,alpha_x,embed);
        
        # Calculate cost
        J = qcf(y,yp,Pz,xu,xp,Pw);
        
        if mfts:
            J += Wx(M,xu,u);
        
        return J,xp,xu,yp,yu;

    # Get parameter gradient
    dth = NG(cost,M.th)/N;
    
    # Update parameters
    th = M.th + alpha_th*dth;

    # Calculate all other variables
    J,xp,xu,yp,yu = cost(M.th);

    return J,th,xp,xu,yp,yu; 

# 1-step algebraical gradient Expectation-Maximization 
def ADEMstep(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed=0,gembed=1,mfts=False,x='def'):
    
    N =  len(u[0,:]);
    th = M.th;

    # Fetch state estimates based on current parameter estimates
    if x=='def':
        xp,xu,yp,yu = FEF(M,x0,u,y,p,alpha_x,embed);
    else: 
        D = shiftmatrix(p=p,embed=embed,nx = int(M.nx/p),dt=M.dt);
        xu = zeros((M.nx,N));
        xp = zeros((M.nx,N));
        yu = zeros((M.ny,N));
        yp = zeros((M.ny,N));
        for k in range(1,N):
            xu[:,k] = dotn(D,x[:,k-1]);
            xp[:,k] = M.f( x[:,k-1],u[:,k-1],th);
            yu[:,k] = M.h( x[:,k  ],u[:,k  ],th);
            yp[:,k] = M.h(xp[:,k  ],u[:,k  ],th);
    
    # Initialize gradients
    dth = zeros((M.nth,1));
    dxdth = zeros((M.nx,M.nth));
    dydth = zeros((M.ny,M.nth));
    ex = zeros((M.nx,1));
    ey = zeros((M.ny,1));
    
    
    # Pre-calculate all matrices (trade storage for efficiency)
    F = zeros((M.nx,M.nth,N));
    H = zeros((M.ny,M.nth,N));
    A = zeros((M.nx,M.nx,N));
    B = zeros((M.nx,M.nu,N));
    C = zeros((M.ny,M.nx,N));
    D = zeros((M.ny,M.nu,N));
    
    if x=='def':
        for k in range(0,N):
            F[:,:,k] = M.F(xu[:,k],u[:,k],th);
            H[:,:,k] = M.H(xu[:,k],u[:,k],th);
            A[:,:,k] = M.A(xu[:,k],u[:,k],th);
            B[:,:,k] = M.B(xu[:,k],u[:,k],th);
            C[:,:,k] = M.C(xu[:,k],u[:,k],th);
            D[:,:,k] = M.D(xu[:,k],u[:,k],th);
    else:
        for k in range(0,N):
            F[:,:,k] = M.F(x[:,k],u[:,k],th);
            H[:,:,k] = M.H(x[:,k],u[:,k],th);
            A[:,:,k] = M.A(x[:,k],u[:,k],th);
            B[:,:,k] = M.B(x[:,k],u[:,k],th);
            C[:,:,k] = M.C(x[:,k],u[:,k],th);
            D[:,:,k] = M.D(x[:,k],u[:,k],th);
            
    # Calculate FE-to-parameter gradient
    for k in range(0,N):
        
        # Errors
        ex[:,0] = xu[:,k] - xp[:,k];
        ey[:,0] = y[:,k]  - yu[:,k];
        
        # To do: Should at some point it turn out that gradient-emnedding is 
        # necessary, then:
#        # State-to-parameter gradients
#        dxdth[:,:] = 0;
#        imax = min(k,gembed);
#        for i in range(1,imax):
#            dxdth[:] = F[:,:,k-imax+i] + dotn(A[:,:,k-imax+i],dxdth)
#        dydth[:,:] = - H[:,:,k] - dotn(C[:,:,k],dxdth);
#        dxdth[:,:] = - F[:,:,k] - dotn(A[:,:,k],dxdth);

        # But until then: 
        dydth[:,:] = - H[:,:,k]; #- dotn(C[:,:,k],F[:,:,k-1]);
        dxdth[:,:] = - F[:,:,k]; #- dotn(A[:,:,k],F[:,:,k-1]);
        
        # update gradient
        dth[:,:] += dotn(tp(dxdth),Pw,ex) + dotn(tp(dydth),Pz,ey);
    
    # Calculate cost
    J = qcf(y,yu,Pz,xu,xp,Pw);
    
    # Update parameters
    th = M.th - alpha_th*dth[:,0];

    return J,th,xp,xu,yp,yu; 

# 1-step algebraical gradient Expectation-Maximization 
def ODEMstep(M,x0,u,y,p,Pw,Pz,alpha_x,alpha_th,embed=0,gembed=1,mfts=False):
    
    th = M.th;

    # Fetch state estimates based on current parameter estimates
    xp,xu,yp,yu = FEFstep(M,x0,u,y,p,alpha_x,embed,mfts);

    # parameter gradients
    dydth = - M.H(xu,u[:,1],th) - dotn(M.C(xu,u[:,1],th),M.F(xu,u[:,1],th));
    dxdth = - M.F(xu,u[:,1],th) - dotn(M.A(xu,u[:,1],th),M.F(xu,u[:,0],th));
        
    # Update gradient
    dth = dotn(tp(dxdth),Pw,xu - xp) + dotn(tp(dydth),Pz,y  - yp);
    
    #Mean-field terms
    if mfts:
        dth += dWxdth(M,xu,u[:,1]);
    
    # Calculate cost
    J = qcf(y,yp,Pz,xu,xp,Pw);
    
    # Update parameters
    th = M.th - alpha_th*dth;

    return J,th,xp,xu,yp,yu; 

#%% Filtering

# Kalman-Filter
def KF(M,x0,u,y,th=[],Q=[],R=[]):
    
    #if M.typ != 'ltiss':
    #    raise Exception('KF code for non-linear state-space (EKF) not complete');
    
    if isempty(th):  
        th = M.th;
    if isempty(Q):  
        Q = M.Q;
    if isempty(R):  
        R = M.R;
        
    A = M.A(zeros(M.nx),zeros(M.nu),th);
    C = M.C(zeros(M.nx),zeros(M.nu),th);
    
    #K,P,S = dare(A,C,Q,R);
    
    nx = M.nx;
    ny = M.ny;
    N =  len(u[0,:]);
    
    K = zeros((nx,ny));
    P = zeros((nx,nx));
    S = zeros((ny,ny));
    
    xp = zeros((nx,N)); xp[:,0] = x0;
    xu = zeros((nx,N)); xu[:,0] = x0;
    yp = zeros((ny,N)); yp[:,0] = M.h(xp[:,0],u[:,0],th);
    yu = zeros((ny,N)); yu[:,0] = M.h(xu[:,0],u[:,0],th);
    
    for k in range(1,N):
        
        A[:,:] = M.A(xu[:,k-1],u[:,k-1],th);
        C[:,:] = M.C(xu[:,k-1],u[:,k-1],th);
        
        # Predict
        xp[:,k] = M.f(xu[:,k-1],u[:,k-1],th);
        yp[:,k] = M.h(xp[:,k  ],u[:,k  ],th);
        P[:,:]  = dotn(A,P,tp(A)) + M.Q;
        
        # Update
        S[:,:]  = dotn(C,P,tp(C)) + M.R;
        K[:,:]  = dotn(P,tp(C),inv(S));
        xu[:,k] = xp[:,k]    + dotn(K,y[:,k] - yp[:,k]);
        yu[:,k] = M.h(xu[:,k  ],u[:,k  ],th);
        P[:,:]  = dotn(eye(M.nx)-dotn(K,C),P);
        
    return K,P,S,xp,xu,yp,yu;

# Online Kalman-Filter step
def KFstep(M,xu,u,y,P):
      
    th = M.th;
        
    A = M.A(xu,u[:,0],th);
    C = M.C(xu,u[:,0],th);

    # Predict
    xp       = M.f(xu,u[:,0],th);
    yp       = M.h(xp,u[:,1],th);
    P        = dotn(A,P,tp(A)) + M.Q;
        
    # Update
    S        = dotn(C,P,tp(C)) + M.R;
    K        = dotn(P,tp(C),inv(S));
    xu       = xp + dotn(K,y - yp);
    P        = dotn(eye(M.nx)-dotn(K,C),P);
    yu       = M.h(xu,u[:,1],th);
    
    return K,P,S,xp,xu,yp,yu;

# Free-Energy Filter
def FEF(M_t,x0,u_t,y_t,p,alpha,embed=0,mfts=False):
    
    # Embed
    #   0   embedded predictions
    #   1   embedded derivatives
    #   2   embedded history 
    
    N = len(u_t[0,:]);
    
    # Fetch functions
    A = M_t.A;
    C = M_t.C;
    f = M_t.f;
    h = M_t.h;
    
    # Initialize storage
    xu_t = zeros((M_t.nx,N)); xu_t[:,0] = x0;
    xp_t = zeros((M_t.nx,N)); xp_t[:,0] = x0;
    yu_t = zeros((M_t.nu,N)); yu_t[:,0] = h(x0, u_t[:,0], M_t.th);
    yp_t = zeros((M_t.ny,N)); yp_t[:,0] = h(x0, u_t[:,0], M_t.th);

    # upper-shift matrix 
    Du = shiftmatrix(p=p,embed=embed,nx = int(M_t.nx/p),dt=1e-2);
    
    #Fetch precisions
    Pw = M_t.Q;
    Pz = M_t.R;
    
    # Run filter
    for k in range(1,N):
        
        # Predictions
        xp_t[:,k] = f(xu_t[:,k-1], u_t[:,k-1], M_t.th);
        yp_t[:,k] = h(xp_t[:,k], u_t[:,k], M_t.th);
        xu_t[:,k] = dotn(Du,xu_t[:,k-1]);
        
        # Errors
        ex        = xu_t[:,k] - xp_t[:,k];
        ey        =  y_t[:,k] - yp_t[:,k];
        
        # Error gradients
        dexdx     = Du - A(xu_t[:,k-1], u_t[:,k-1], M_t.th);
        deydx     =    - C(xu_t[:,k], u_t[:,k], M_t.th);

        # Free-Energy gradient
        dFEdx    = dotn(tp(dexdx),Pw,ex) + dotn(tp(deydx),Pz,ey);
        
        # Mean-field terms if specified
        if mfts:
            dFEdx += dWthdx(M_t,xu_t[:,k-1],u_t[:,k-1]);
        
        # Upddates
        xu_t[:,k] = xu_t[:,k] - alpha*dFEdx
        yu_t[:,k] = h(xu_t[:,k], u_t[:,k], M_t.th);
    
    return xu_t,xp_t,yu_t,yp_t; 

def shiftmatrix(dt=1e-2,nx=2,p=1,embed=0):
    if embed == 1 or embed == 0:
        Du = kron(eye(p,p,1),eye(nx));
        if embed == 1:  
            Du = expm(Du*dt);
    elif embed == 2:
        Du = kron(eye(p,p,-1),eye(nx));
    return Du;


# # Online Free-Energy Filter step
def FEFstep(M_t,x0,u_t,y_t,p,alpha,embed=0,mfts=False):
    
    # Embed
    #   0   embedded predictions
    #   1   embedded derivatives
    #   2   embedded history 
    
    # Fetch matrices
    A = M_t.A(x0, u_t[0], M_t.th);
    C = M_t.C(x0, u_t[0], M_t.th);

    # upper-shift matrix 
    if embed == 1 or embed == 0:
        Du = kron(eye(p,p,1),eye(int(M_t.nx/p)));
        if embed == 1:  
            Du = expm(Du*M_t.dt);
    elif embed == 2:
        Du = kron(eye(p,p,-1),eye(int(M_t.nx/p)));
    
    #Fetch precisions
    Pw = M_t.Q;
    Pz = M_t.R;
        
    # Predictions
    xp_t = M_t.f(x0,   u_t[:,0], M_t.th);
    yp_t = M_t.h(xp_t, u_t[:,1], M_t.th);
    xu_t = dotn(Du,x0);

    # Free-Energy gradient
    dFEdx = dotn(tp(Du - A),Pw,xu_t - xp_t) + dotn(tp(- C),Pz,y_t - yp_t);
        
    # Mean-field terms if specified
    if mfts:
        dFEdx += dWthdx(M_t,x0,u_t[:,0]);
        
    # Upddates
    xu_t = xu_t - alpha*dFEdx
    yu_t = M_t.h(xu_t, u_t[:,1], M_t.th);
    
    return xu_t,xp_t,yu_t,yp_t; 

#%% System generalization
    
# Generalize Model
def generalize(M,p,S=[]):
    
    # Fetch everything from model
    f    = M.f;
    A    = M.A;
    B    = M.B;
    F    = M.F;
    h    = M.h;
    C    = M.C;
    D    = M.D;
    H    = M.H;
    th   = M.th;
    nx   = M.nx
    nu   = M.nu;
    ny   = M.ny;
    nth  = M.nth;
    name = M.name;
    typ  = M.typ;
    dt   = M.dt;
    Q    = M.Q;
    R    = M.R;
    
    # Generalize all system functions and matrices
    # State transition function 
    def f_t(x,u,th):
        x_t = zeros(p*nx);
        for i in range(0,p):
            x_t[i*nx:(i+1)*nx] = f(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return x_t;
    
    # State transition function to state gradient
    def A_t(x,u,th):
        A_t = zeros((p*nx,p*nx));
        for i in range(0,p):
            A_t[i*nx:(i+1)*nx,i*nx:(i+1)*nx] = \
                            A(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return A_t;
    
    # State transition function to input gradient
    def B_t(x,u,th):
        B_t = zeros((p*nx,p*nu));
        for i in range(0,p):
            B_t[i*nx:(i+1)*nx,i*nu:(i+1)*nu] = \
                            B(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return B_t;
    
    # State transition function to parameter gradient
    def F_t(x,u,th):
        F_t = zeros((p*nx,nth));
        for i in range(0,p):
            F_t[i*nx:(i+1)*nx,:] = \
                            F(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return F_t;
    
    # Output function 
    def h_t(x,u,th):
        y_t = zeros(p*ny);
        for i in range(0,p):
            y_t[i*nx:(i+1)*nx] = h(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return y_t;
    
    # Output function to state gradient
    def C_t(x,u,th):
        C_t = zeros((p*ny,p*nx));
        for i in range(0,p):
            C_t[i*ny:(i+1)*ny,i*nx:(i+1)*nx] = \
                            C(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return C_t;
    
    # Output function to input gradient
    def D_t(x,u,th):
        D_t = zeros((p*ny,p*nu));
        for i in range(0,p):
            D_t[i*ny:(i+1)*ny,i*nu:(i+1)*nu] = \
                            D(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return D_t;
    
    # Measurement function to parameter gradient
    def H_t(x,u,th):
        H_t = zeros((p*ny,nth));
        for i in range(0,p):
            H_t[i*ny:(i+1)*ny,:] = \
                H(x[i*nx:(i+1)*nx],u[i*nu:(i+1)*nu],th);
        return H_t;
    
    # Include corellation information if specified
    if not isempty(Q) and not isempty(R):
        if not isempty(S):
            Pw = kron(S,inv(Q));
            Pz = kron(S,inv(R));
        else:
            Pw = kron(eye(p),inv(Q));
            Pz = kron(eye(p),inv(R));
    else:
        Pw = [];
        Pz = [];
        
    M_t = Model('Generalized '+ name, typ, dt, [p*nx,p*nu,p*ny,nth], \
                f_t,A_t,B_t,F_t,h_t,C_t,D_t,H_t,th,Pw,Pz);
    
    return M_t, kron(eye(p,p,1),eye(int(M_t.nx/p)));

#%% Mean-field-term gradients
def dWthdx(M_t,x,u):
    return zeros(M_t.nx);

def dWxdth(M_t,x,u):
    return zeros(M_t.nth);

def Wx(M,x,u):
    return 0;

#%% Cost-functions

# Quadratic cost function   
def qcf(y,yp,invR=[],xu=[],xp=[],invQ=[]):
    
    if invR == []: invR = eye(len(y)); 
    if invQ == []: invQ = eye(len(xu));   
    
    try:
        N = len(y[0,:]);
    except: 
        N = 1;
    if N == 1:
        J = dot(transpose(y-yp),dot(invR,y-yp));
    
        if all([len(xu),len(xp),len(invQ)]) != 0:
            J += dot(transpose(xu-xp),dot(invQ,xu-xp));
    else:
        J = 0;
        for i in range(0,N):
            J += dot(transpose(y[:,i]-yp[:,i]),dot(invR,y[:,i]-yp[:,i]));
    
        if all([len(xu),len(xp),len(invQ)]) != 0:
            for i in range(0,N):
                J += dot(transpose(xu[:,i]-xp[:,i]),dot(invQ,xu[:,i]-xp[:,i]));
    return J;

# Log-Likelihood
def ll(y,yp,R,xu,xp,Q): 
    try:
        N = len(y[0,:]);
    except: 
        N = 1;
    
    J = - N*(log(norm(Q,2))+log(norm(R,2)));     
    
    J += -qcf(y,yp,inv(R),xu,xp,inv(Q)) ;    
    
    return J;


#%% Temporal Corellation

# Temporal Correllation matrix for embedded derivatives
def TCM(iden,n,s):
    S = zeros((n,n));
    
    # Gaussian Filter case
    if iden=='GF':
        rho = zeros(2*n);
        for i in range(0,2*n):
            if i % 2 == 0:
                rho[i] = fct(i)/((-2)**int(i/2)*fct(i/2)*s**(i));
            else:
                pass;
        for i in range(0,n):
            S[i,:] = (-1)**i*rho[i:i+n]
        
        S = inv(S);
    else:
        raise Exception('TCM-code for non-Gaussian auto-corellation\
                        not complete');
    return S;

def rho(k,s):
    return exp(-0.5*k**2/s**2);

# Temporal Correllation matrix for embedded predictions (future or past)
def DTCM(iden,p,s):
    S = zeros((p,p));
    if iden=='GF':
        for i in range(0,p):
            for j in range(0,p):
                S[i,j] = rho(i,s)*rho(j,s);
    else:
        raise Exception('DTCM-code for non-Gaussian auto-corellation\
                        not complete');
                
    return S;


#%% Signal differentiation/prediction

# Numerical causal derivative observer (causal but not accurate)
def NDO(k,x0,dt):
    
    N = len(x0[0,:]);
    x = zeros((k,N));
    x[0,:] = x0[0,:];
    
    for j in range(1,k):
        x[j,j:N] = x0[:,0:N-j];
    
    P = dotn(powdiag(k,1/dt),pascal(k),powdiag(k,-1));
    x = dot(P,x);
    return x;

# Numerical causal predictions observer (causal but not accurate)
def NPO(k,x0,dt):
    
    N = len(x0[0,:]);
    x = zeros((k,N));
    x[0,:] = x0[0,:];
    
    for j in range(1,k):
        x[j,j:N] = x0[0,0:N-j];
    
    P = dotn(pascal(k),pascal(k),powdiag(k,-1));
    x = dot(P,x);

    return x;

# Numerical non-causal derivative observer (non-causal but accurate, to be used in offline dem)
def NCNDO(p,u,dt):
    try: 
        N = len(u[0,:]);        
        x = zeros((p,N));
        x[0,:] = u[0,:];
    except:
        N = len(u);        
        x = zeros((p,N));
        x[0,:] = u;
        
        
    for i in range(1,p):
        for k in range(i,N-i):
            x[i,k] = (x[i-1,k+1] - x[i-1,k-1])/dt;
    return x;

# Stable causal derivative observer (causal and accurate, but sensitive to initial conditions)
def SDO(p,u,x0,dt,wc=100): 
        
        p +=1;
    
        N = len(u[0,:]);
             
        P = pascal(p+1);
        A = eye(p,p,1);
        a = zeros(p)
        for i in range(0,p):
            a[p-1-i] = P[-1,i+1]*wc**(i+1);
        A[-1,:] = -a;
        B = zeros((p,1));
        B[-1,-1] = 1;
        C = wc**p*eye(p-1,p);
        D = zeros((p-1,1));
        
        Ad = expm(A*dt);
        Bd = dotn(inv(A),Ad-eye(p),B);
        
        x = zeros((p,N));
        y = zeros((p-1,N));
        x[0:p-1,0] = x0;
        y[:,0] = dotn(C,x[:,0])    +  dotn(D,u[:,0]);
        for k in range(1,N):
            x[:,k] = dotn(Ad,x[:,k-1]) +  dotn(Bd,u[:,k-1]);
            y[:,k] = dotn(C,x[:,k])    +  dotn(D,u[:,k]);
        return y;
        
        
        
    