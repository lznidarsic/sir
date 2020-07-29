from numpy  import zeros, sin, cos, log10, pi,linspace,logspace, sqrt, exp

# Step, ramp, nramp
def nramp(t,t1,a,n,t2):
    N = len(t);
    x = zeros(N);
    for i in range(0,N):
        if t[i] > t1 and t[i] <= t2:
            x[i] = a*(t[i]-t1)**n;
        elif t[i] > t2:
            x[i] = a*(t2-t1)**n;
    return x;

# sweep, linsweep, logsweep
def nsweep(t,f1,f2,a1,a2,fres,ares):
    N = len(t);
    x = zeros(N);
    if fres == 'log':
        f = logspace(log10(f1),log10(f2),N);
    else:
        f = linspace(f1,f2,N);
        
    if ares == 'log':
        a = logspace(log10(a1),log10(a2),N);
    else:
        a = linspace(a1,a2,N);  
    for i in range(0,N):
        x[i] = a[i]*sin(2*pi*f[i]*t[i]);
    return x;

# sinsum, linsinsum, logsinsum
def sinsum(t,n,f1,f2,a1,a2,fres,ares):
    
    x = zeros(len(t));
    if fres == 'log':
        f = logspace(log10(f1),log10(f2),n);
    else:
        f = linspace(f1,f2,n);
        
    if ares == 'log':
        a = logspace(log10(a1),log10(a2),n);
    else:
        a = 1/n*linspace(a1,a2,n);  
        
    for i in range(0,n):
        x += a[i]*sin(2*pi*f[i]*t);
    return x;

# sinsum, linsinsum, logsinsum
def cossum(t,n,f1,f2,a1,a2,fres,ares):
    
    x = zeros(len(t));
    if fres == 'log':
        f = logspace(log10(f1),log10(f2),n);
    else:
        f = linspace(f1,f2,n);
        
    if ares == 'log':
        a = logspace(log10(a1),log10(a2),n);
    else:
        a = 1/n*linspace(a1,a2,n);  
        
    for i in range(0,n):
        x += a[i]*cos(2*pi*f[i]*t);
    return x;




def setdim(p,n,dflt):
    try: 
        l = len(p);
    except:
        l = 1;
    if l == n:
        pass;
    elif l < n:
        p=dflt;
    elif l > n:
        p = p[0:n];
    return p;

def getSignal(ident,t,p1=[],p2=[],p3=[],p4=[],p5=[]):
    if ident == "STEP":
        t1 = setdim(p1,1,0);
        t2 = setdim(p2,1,t[-1]);
        a  = setdim(p3,1,1);
        return nramp(t,t1,a,0,t[-1]) - nramp(t,t2,a,0,t[-1]);  
    elif ident == "RAMP":
        t1 = setdim(p1,1,0);
        t2 = setdim(p2,1,t[-1]);
        a  = setdim(p3,1,1);
        return nramp(t,t1,a,1,t2);
    elif ident == "NRAMP":
        t1 = setdim(p1,1,0);
        t2 = setdim(p2,1,t[-1]);
        a  = setdim(p3,1,1);
        n  = setdim(p4,1,2);
        return nramp(t,t1,a,n,t2);
    elif ident == "SWEEP":
        f1 = setdim(p1,1,1);
        f2 = setdim(p2,1,10);
        A  = setdim(p3,1,1);
        return nsweep(t,f1,f2,A,A,'lin','lin');
    elif ident == "LINSWEEP":
        f1 = setdim(p1,1,1);
        f2 = setdim(p2,1,10);
        A1 = setdim(p3,1,1);
        A2 = setdim(p4,1,0);
        return nsweep(t,f1,f2,A1,A2,'lin','lin');
    elif ident == "LOGSWEEP":
        f1 = setdim(p1,1,1);
        f2 = setdim(p2,1,10);
        A1 = setdim(p3,1,1);
        A2 = setdim(p4,1,1e-3);
        return nsweep(t,f1,f2,A1,A2,'log','log');
    elif ident == "SINSUM":
        n  = setdim(p1,1,10);
        f1 = setdim(p2,1,1);
        f2 = setdim(p3,1,10);
        A  = setdim(p4,1,1);
        return sinsum(t,n,f1,f2,A,A,'lin','lin');
    elif ident == "LINSINSUM":
        n  = setdim(p1,1,10);
        f1 = setdim(p2,1,1);
        f2 = setdim(p3,1,10);
        A1 = setdim(p4,1,1);
        A2 = setdim(p4,1,0);
        return sinsum(t,n,f1,f2,A1,A2,'lin','lin');
    elif ident == "LOGSINSUM":
        n  = setdim(p1,1,10);
        f1 = setdim(p2,1,1);
        f2 = setdim(p3,1,10);
        A1 = setdim(p4,1,1);
        A2 = setdim(p5,1,1e-3);
        return sinsum(t,n,f1,f2,A1,A2,'log','log');
    elif ident == "DLOGSINSUM":
        n  = setdim(p1,1,10);
        f1 = setdim(p2,1,1);
        f2 = setdim(p3,1,10);
        A1 = setdim(p4,1,1);
        for i in range(0,len(A1)):
            A1[i] = A1[i]*f1[i];
            
        A2 = setdim(p5,1,1e-3);
        return cossum(t,n,f1,f2,A1,A2,'log','log');
    elif ident == "DDLOGSINSUM":
        n  = setdim(p1,1,10);
        f1 = setdim(p2,1,1);
        f2 = setdim(p3,1,10);
        A1 = setdim(p4,1,1);
        A2 = setdim(p5,1,1e-3);
        return sinsum(t,n,f1,f2,A1,A2,'log','log');
    elif ident=="GAUSSIAN":
        A = p1;
        s = p2;
        t0 = p3;
        alpha = -0.5*((t-t0)/s)**2;
        return A*exp(alpha)/(s*sqrt(2*pi));
#    elif ident=="DGAUSSIAN":
#        A = p1;
#        s = p2;
#        t0 = p3;
#        alpha = -0.5*((t-t0)/s)**2;
#        return A*exp(alpha)/(s*sqrt(2*pi));
#    elif ident=="DDGAUSSIAN":
#        A = p1;
#        s = p2;
#        t0 = p3;
#        alpha = -0.5*((t-t0)/s)**2;
#        return A*exp(alpha)/(s*sqrt(2*pi));
    elif ident=="SUMGAUSSIAN":
        A = p1;
        s = p2;
        t0 = p3;
        res = 0;
        for i in range(len(A)):
            alpha = -0.5*((t-t0[i])/s[i])**2;
            res+= A[i]*exp(alpha)/(s[i]*sqrt(2*pi))
        return res;
    else:
        print("Unknown system identifier. Options are: "+\
              "\n   STEP       Step input" +\
              "\n   RAMP       Ramp input" +\
              "\n   nRAMP      n-th order Ramp input" +\
              "\n   SWEEP      frequency sweep" +\
              "\n   LINSWEEP   Frequency sweep with linear amplitude slope" +\
              "\n   LOGSWEEP   Frequency sweep with logarthmic frequency "+\
                              "and amplitude slope" +\
              "\n   SUMSIN     Linear slope frequency constant amplitude sum of sines"  +\
              "\n   LINSUMSIN  Linear slope frequency and amplitude sum of sines\n"  +\
              "\n   LOGSUMSIN  Logarithmic slope frequency and amplitude sum of sines\n" );
              
