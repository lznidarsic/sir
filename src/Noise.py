from numpy import zeros, ones, pi, linspace, log10, random,sinc 
from numpy import reshape, exp, sqrt,amax,fft,mean,std
import matplotlib.pyplot as p
from src.Timer import Timer
from src.myfunctions import isempty

class Noise:
    """
    Initialization inputs
    Name   def. val   description
    dt     1e-3       Sampling period (seconds)
    T      10         Length of time-sequence (seconds)
    d      1          Distribution specifier def = 1
                          1 = Gausian
                          2 = Uniform
    f      1          Frequency spectrum color
                          1 = White      0  dB/dec
                          2 = Pink     -10  dB/dec
                          3 = Red      -20  dB/dec
                          4 = Blue      10  dB/dec
                          5 = Violet    20  dB/dec
                          6 = f-alpha   x   dB/dec
    a        1        Autocorrelation function
                          1 = None
                          2 = Causal integration
                          3 = Gaussian
                          4 = Block
    pd      [0,1]     Parameters accompanying chosen
                      distribution (n x np_d)
    pf      []        Parameters accompanying chosen
                      frequency characteristics (1 x np_f)
    pa      []        Parameters accompanying chosen
                      autocorellation function (1 x np_a)
    seed    []        For reproducible random noise\n
    Note : to get Brownian noise, choose White and Integration
    To obtain the time and noise sequence use getNoise()
    To visualize the time and noise sequence use showNoise()
    """
    
    __author__    = 'Laurens Y.D. Žnidaršič';
    __title__     = 'Noise generator';
    __version__   = '1.0.0';
    __license__   = 'Free to use';
    __copyright__ = 'Copyright 2020 by Laurens Y.D. Žnidaršič';
    
    __slots__ =  ['_dname','_fname','_aname','_panames','_pdnames','_pfnames',\
                  '_dt','_d','_f','_a','_pd','_pf','_pa','_k','_w','_t'];
                 
    def __init__(self,dt=1e-3,T=10,d=1,f=1,a=1,pd=[],pf=0,pa=[],seed=[],prnt=False):
        
        self._pf = pf;
        self._dt = dt;
        self._t = linspace(0,T,int(T/dt+1))
        self.__setDistribution(d,pd);
        self.__setFrequencies(f,pf);
        self.__setCorrelation(a,pa);
        self.__generateSequence(T,seed);
        
        if prnt:
            # Print statement
            print("Created: sequence of " + self.__str__());
        return;
        
    def __str__(self):
        ret = self.__repr__() + "\n";
        
        ret += "Distribution characteristics:\n Function: " + \
        self._dname + "\n"
        for i in range(0,len(self._pdnames)):
            p = "(%.2f" % self._pd[0,i];
            for j in range(1,len(self._pd[:,0])):
                p += ",%.2f" % self._pd[j,i]
            ret += " " + self._pdnames[i] + ": "+ p +")\n";
        ret += "Frequency characteristics:\n Color: " + \
        self._dname + "\n";
        p = "%.d" % self._pf
        ret += " " + self._pfnames[0] + ": "+ p +" dB/dec\n";
        
        ret +=  "Auto-correlation characteristics:\n Function: " + \
        self._dname + "\n";
        for i in range(0,len(self._panames)):
            ret += " " + self._panames[0] + ": %.1f" % self._pa +"\n";
        
        ret += "Time characteristics:"
        ret += "\n T  " + "%.2f (s)" % self._t[len(self._t)-1];
        ret += "\n dt " + "1e%d (s)\n" % log10(self._dt);
        
        return ret;
        
    def __repr__(self):
        if self._a == 1:
             ret = self._dname + ' ' + self._fname ;
        elif self._a == 2 and self._f == 1:
            ret = 'Brownian ';
        else:
            ret = self._aname + ' ' + self._dname + ' ' + self._fname ;
        ret += ' Noise';
        return  ret;
        
    def __len__(self):
        return len(self._w[:,0]);
    
    def __type__(self):
        print('noise generator object');
        return;
    
    def getNoise(self):
        return self._t, self._w;

    def getData(self,kcor='def',twin='def'):
        
        if kcor !='def':
            self._k = kcor;
        w = self._w;   
        t = self._t;
        n = len(w[:,0]);           
        
        p.clf();
        xs,ys,fs,As,xcs,ycs = self.__getBaselines();

        lim = [xs[0,0],xs[0,-1]];
        x,y = self.__getProbabilityDistribution(w,5e-2,lim);
        f,A = self.__myFFT(w,t,1);
        xc,yc = self.__getAutoCorrelation(w);
        #print(n)
        
        # Rescale Amplitude baseline
        for i in range(0,n):
            As[i,:] = mean(A[i,:])*As[i,:]/mean(As[i,:]); 
        if self._a == 3:
            fs = fs[0:int(1/(self._pa))];
            As = As[:,0:int(1/(self._pa))];  
            
        if twin!='def':
            w = w[:,0:int(twin/self._dt+1)];
            t = t[  0:int(twin/self._dt+1)];

                
        return t,w,x,xs,y,ys,f,fs,A,As,xc,xcs,yc,ycs;
    
    def getParams(self):
        return self._pd,self._k,len(self._w[:,0]);
    
    def __generateSequence(self,T,seed=[]):
        
        # Set noise reproducibility seed
        if not isempty(seed):
            random.seed(seed);
            
        # Initialize sizes and data storage
        n = len(self._pd[:,0]);
        N = int(T/self._dt+1);
        wout = zeros((n,N));
        
        minsamps = int(1e6);
        
        # Generate random sequences (with white frequency)
        if self._d == 1: 
            w = random.normal(0,1,n*max([N,minsamps]));                
        elif self._d == 2: 
            w = random.uniform(-0.5,0.5,n*max([N,minsamps]));
        w = reshape(w,(n,max([N,minsamps])));

        # Include frequency characcteristics into sequences
        for i in range(0,n):
            if self._f == 1: # ... with 1 frequency
                wout[i,:] = w[i,0:N]; # sequence is already correct;
            elif self._f >= 2 and self._f <= 6:
                t_temp = linspace(0,max([N,minsamps])*self._dt, max([N,minsamps]));
                f = fft.rfftfreq(t_temp.shape[-1],self._dt)*2;
                A = fft.rfft(w[i,:]);
                for j in range(0,len(f)):
                    if f[j] != 0: A[j] = A[j]*(f[j]**(self._pf/20));
                w[i,:] = fft.irfft(A);
#                f,A = self.__myFFT(w[i,:],t_temp,1);
#                w[i,:] = fft.irfft(A);
                # Shorten sequence
                wout[i,:] = w[i,0:N];   
        
        # Include auto-correlation characteristics into sequences
        if self._a == 1: # No autocorrellation
            pass; 
        elif self._a == 2: # Causal integration
            for i in range(1,N):
                wout[:,i] = wout[:,i-1] + self._dt*wout[:,i]
        elif self._a == 3: # Gaussian auto-correlation
            wout = self.__correlateGaussian(wout);
        elif self._a == 4: # block auto-correlation
            wout = self.__correlateBlock(wout);
        
        # Rescale
        for i in range(0,n):
            wout[i,:] = wout[i,:]-mean(wout[i,:]); 
            if self._d == 1: 
                wout[i,:] = wout[i,:]/std(wout[i,:]);
            elif self._d == 2: 
                wout[i,:] = wout[i,:]/(max(wout[i,:])-min(wout[i,:]));
            wout[i,:] = self._pd[i,1]*wout[i,:] + self._pd[i,0];
            
        self._w = wout;
        return 0;

    def __correlateGaussian(self,w):
        N = len(w[0,:]);
        n = len(w[:,0]);
        wout = zeros((n,N));
        tol = 1e-6;
        # find for which coordinate the gaussian < tol
        gaus = self.__gaussian(self._t,0,self._pa);
        k=0
        while gaus[k] > tol:
            k+=1;
        gaus = zeros(2*k);
        T = Timer("Convolving signal with Gaussian..",0,N)
        for j in range(0,N):
            kmin = max([0,j-k]);
            kmax = min([N-1,j+k]);
            if j <= k or j >= N-k:
                gaus = self.__gaussian(self._t[kmin:kmax],self._t[j],self._pa);
            else:
                gaus[:] = self.__gaussian(self._t[kmin:kmax],self._t[j],self._pa);
            for i in range(0,n):
                wout[i,j] = sum(gaus*w[i,kmin:kmax]);
            T.looptime(j)
        return wout;
    
    def __correlateBlock(self,w):
        N = len(w[0,:]);
        n = len(w[:,0]);
        wout = zeros((n,N));
        T = Timer("Convolving signal with Block..",0,N)
        for j in range(0,N):
            block = self.__block(self._t,self._t[j],self._pa);
            for i in range(0,n):
                
                wout[i,j] = sum(block*w[i,:]);
            T.looptime(j)
        return wout;
        
    def __block(self,t,t1,tw):
        x = zeros((len(t)));
        x[(t>t1-tw/2) & (t<t1+tw/2)] = 1/tw;
        return x;
        
        
    
    def __setCorrelation(self,a,pa=[]):
        if a > 0 and a <= 4:
             self._a = a;
        else:
            self._a = 1;
            print("WARNING: Autocorellation specifier exceeds number "+\
                  "of options\n Setting default (1)");
        if self._a == 1: # No autocorrellation
             self._aname = "None";
             self._panames = [];
             self._k = 100;
             self._pa = 0;
        elif self._a == 2: # (Causally) integrated noise
             self._aname = "Causally integrated";
             self._panames = [];
             self._k = int(1e-1*self._t[len(self._t)-1]/self._dt);
             self._pa = 0;
        elif self._a == 3: # Gaussian autocorrelated noise
             self._aname = "Gaussian convolved";
             if len(pa) == 1:
                 self._pa = pa[0];
             else:
                 self._pa = 1;
             self._k = int(4*self._pa/self._dt);
             self._panames = ['Standard deviation (s)'];
        elif self._a == 4: # Block autocorrelated noise
             self._aname = "Block convolved";
             if len(pa) == 1:
                 self._pa = pa[0];
             else:
                 self._pa = 1;
             self._k = int(4*self._pa/self._dt);
             self._panames = ['Block width (s)'];
    
    def __setFrequencies(self,f,pf):
        if f > 0 and f <= 6:
            self._f = f;
        else:
            self._f = 1;
            print("WARNING: frequency characteristics specifier exceeds "+\
                  "number of options\n Setting default (1)");
        if self._f == 1 or (self._f == 6 and pf == 0):
            self._pf = 0;
            self._fname = "White"
            self._pfnames = ["Slope"];
        elif self._f == 2:
            self._pf = -10;
            self._fname = "Pink"
            self._pfnames = ["Slope"];
        elif self._f == 3:
            self._pf = -20;
            self._fname = "Red"
            self._pfnames = ["Slope"];
        elif self._f == 4:
            self._pf = 10;
            self._fname = "Blue"
            self._pfnames = ["Slope"];
        elif self._f == 5:
            self._pf = 20;
            self._fname = "Violet"
            self._pfnames = ["Slope"];
        elif self._f == 6:
            self._pf == -pf;
            if self._pf > 0:
                self._fname = "f-alpha"
            else:
                self._fname = "1/f-alpha"
            self._pfnames = ["Slope"];

        return 0;
    
    def __setDistribution(self,d,pd):  
        if len(pd) != 0:
            n =  pd[:,0].size
            np = pd[0,:].size
        else:
            n = 1;
            np = 2;
            
        if d > 0 and d <= 6:
            self._d = d;
        else:
            self._d = 1;
            print("WARNING: frequency characteristics specifier exceeds "+\
                  "number of options\n Setting default (1)");
        if d == 1:
            self._dname = "Gaussian"
            self._pdnames = ["Mean","Standard Deviation"];
            if pd == []:
                self._pd = zeros((1,2));
                self._pd[0,1] = 1;
            elif np == 2:
                self._pd = pd;
            else:
                self._pd = zeros(n,2);
                self._pd[:,1] = ones(n,1);
                print("Number of distribution parameters is "+str(pd[0,:].size)+". \nGaussian needs 2 (mean,std). \nSetting default values (0,1)")
        elif d == 2:
            self._dname = "Uniform"
            self._pdnames = ["Mean"," Width"];
            self._d = 2;
            if pd == []:
                self._pd = zeros((1,2));
                self._pd[0,1] = 1;
            elif np == 2:
                self._pd = pd;
            else:
                self._pd = zeros(n,2);
                self._pd[:,1] = ones(n,1);
                print("Number of distribution parameters is "+str(pd[0,:].size)+". \nUniform needs 2 (mean,width). \nSetting default values (0,1)")       
        else:
            print("Unknown distribution specifier. \nSetting default (Gaussian with (0,1))");
            self._dname = "Gaussian"
            self._pdnames = ["Mean         ", "Standard Deviation"];
            self._pd = zeros(n,2);
            self._pd[:,1] = ones(n,1);
    
    def __getBaselines(self):
        k = self._k;
        n = len(self._pd[:,0]); 
        
        if self._d==1:
            mi = min(self._pd[:,0]-4*self._pd[:,1]);
            ma = max(self._pd[:,0]+4*self._pd[:,1]);
            x = zeros((1,int(1e3+1)));
            x[0,:] = linspace(mi,ma,int(1e3+1));
            y = zeros((n,len(x[0,:])));
            
            for i in range(0,n):
                mu = self._pd[i,0] 
                std = self._pd[i,1]    
                y[i,:] = self.__gaussian(x,mu,std);

        elif self._d==2:
            mi = min(self._pd[:,0]-1.5*self._pd[:,1]);
            ma = max(self._pd[:,0]+1.5*self._pd[:,1]);
            x = zeros((1,9));
           
            y = zeros(n,9); # Needs Fixing!!!
            for i in range(0,n):
                mu = self._pd[i,0] 
                wid = self._pd[i,1]/2
                x[0,:] = [mi,mu-3*wid,mu-wid,mu-wid,mu,mu+wid,mu+wid,mu+3*wid,ma];
                y[i,3:6] = 0.5/wid;
        
        fc = 1.0/self._dt;
        A = zeros((n,1001));
        f = linspace(fc/1001,fc,1001);
        for i in range(0,n): 
            A[i,:] = f**(self._pf/20);
            mn = sum(A[i,:])/len(A[i,:]);
            A[i,:] = self._pd[i,1]*A[i,:]/mn + self._pd[i,0];
                
        if self._a == 1:
            xc = zeros((1,5));
            xc[0,0] = -k;
            xc[0,4] = k;
            yc = zeros((n,5));
            yc[0,2] = 1;
            
        elif self._a == 2:   
            xc = zeros((1,5));
            xc[0,0] = -k;
            xc[0,4] =  k;
            yc = zeros((n,5));
            yc[0,0:2] = 1;
            
            for i in range(0,n): 
                A[i,:] = A[i,:]*f**(-1);
                mn = sum(A[i,:])/len(A[i,:]);
                A[i,:] = self._pd[i,1]*A[i,:]/mn + self._pd[i,0];
            
        elif self._a == 3:   
            N = int(2*self._k+1);
            xc = zeros((1,N));
            yc = zeros((1,N));
            xc[0,:] = linspace(-4*self._pa/self._dt,4*self._pa/self._dt,N);
            yc[0,:] = self.__gaussian(xc,0,self._pa/self._dt);     
            yc[0,:] = yc[0,:]/amax(yc[0,:]);
            
            g = self.__gaussian(f,0,1/(pi*self._pa)); 
            for i in range(0,n): 
                A[i,:] = A[i,:]*g;
                mn = sum(A[i,:])/len(A[i,:]);
                A[i,:] = self._pd[i,1]*A[i,:]/mn + self._pd[i,0];
        
        elif self._a == 4:   
            N = int(2*self._k+1);
            xc = zeros((1,N));
            yc = zeros((1,N));
            xc[0,:] = linspace(-4*self._pa/self._dt,4*self._pa/self._dt,N);
            yc[0,:] = self.__gaussian(xc,0,self._pa/self._dt);     
            yc[0,:] = yc[0,:]/amax(yc[0,:]);
            
            #g = zeros((1,len(f)));
            w = self._pa;
            #d = 80;
            #for i in range(0,len(f)):
                #g[0,i] =  abs(exp(-(f[i]+d)*w/4)*sin(3*(f[i]+d)*w/2)) + 1e-5;
            for i in range(0,n): 
                #A[i,:] = A[i,:]*g;
                A[i,:] = abs(sinc(f*w/2.2))+1e-6; 
                mn = sum(A[i,:])/len(A[i,:]);
                A[i,:] = self._pd[i,1]*A[i,:]/mn + self._pd[i,0]; 
                
            
        return x,y,f,A,xc,yc;
    
    def __getAutoCorrelation(self,w):
        
        k = self._k;
        
        n = len(w[:,0]);
        N = len(w[0,:]);
        x = zeros((1,int(2*k+1)));
        x[0,:] = linspace(-k,k,int(2*k+1));
        y = zeros((n,len(x[0,:])));
    
        for i in range(0,n):
            for j in range(0,int(2*k+1)):
                y[i,j] = sum(w[i,k:N-k]*w[i,j:N-2*k+j]);
            y[i,:] = (y[i,:]-min(y[i,:]))/(max(y[i,:])-min(y[i,:]));
        return x,y;
    
    def __getProbabilityDistribution(self,w,bw=1e-1,lims=[],Nx=1001):
        if lims == []:
            lims = [-1,1];
        n = len(w[:,0]);
        x = zeros((1,Nx));
        x[0,:] = linspace(lims[0],lims[1],Nx);
        Nw = len(w[0,:]);
        y = zeros((n,Nx));
        for i in range(0,n):
            for j in range(0,Nw):
                y[i,:] += self.__gaussian(x[0,:],w[i,j],bw);
            y[i,:] = y[i,:]/(sum(y[i,:])*(x[0,2]-x[0,1]));
        return x,y;    

    def __myFFT(self,w,t,real=1):
        n = len(w[:,0]);
        N = len(w[0,:]);
        f = zeros((1,int(N/2+1)));
        A = zeros((n,int(N/2+1)),"complex");
        for i in range(0,n):
            A[i,:] = fft.rfft(w[i,:]);
        f[0,:] = fft.rfftfreq(t.shape[-1],self._dt)*2;
        if real == 1:
            A = abs(A);
        return f,A;
    
    def __gaussian(self,x,mu,std):
        return exp(-0.5*((x-mu)/std)**2)/(std*sqrt(2*pi));

