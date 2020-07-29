import matplotlib.pyplot as plt
from numpy import log10, zeros, linspace, ceil,amax,amin,eye
from numpy.linalg import inv, norm
#from src.parameterestimation import EMstep
from src.Timer import Timer
#from src.Noise import Noise
from src.myfunctions import isempty
from src.pedata import pedata;

defcols = ["aquamarine","turquoise",\
           "lightcoral","indianred",\
           "palegreen" ,"springgreen",\
           "khaki" ,"goldenrod"];
           

#tudcols  = [(30.0/255.0, 97.0/255.0,136.0/255.0),\
#            (69.0/255.0, 69.0/255.0, 69.0/255.0),\
#            ( 0.0/255.0,166.0/255.0,214.0/255.0),\
#            (34.0/255.0, 34.0/255.0, 34.0/255.0)];

tudcols  = [(  0.0/255.0, 166.0/255.0, 214.0/255.0),\
            (165.0/255.0, 202.0/255.0,  26.0/255.0),\
            (231.0/255.0,  76.0/255.0,  16.0/255.0),\
            ( 56.0/255.0,  41.0/255.0, 118.0/255.0),\
            ( 30.0/255.0,  97.0/255.0, 136.0/255.0),\
            ( 69.0/255.0,  69.0/255.0,  69.0/255.0),\
            (  0.0/255.0, 166.0/255.0, 214.0/255.0),\
            ( 34.0/255.0,  34.0/255.0,  34.0/255.0)];
 
 
def plotcost(J, maxth,col,nrows = 1,irow=1,styl='-'):
    nth = len(J[:,0]);
    res = len(J[0,:]);
    
    xtx =      [-0.9*maxth,0,0.9*maxth];
    w = (1-4*0.01)/3;
    h =  (1-nrows*0.01-0.08)/nrows;
    theta = linspace(-maxth,maxth,res);
    
    for i in range(0,nth):
        # Plot current costs + some plot-specific figure settings
        thi = '\u03b8$_' + str(i+1) + '$';
        if irow == nrows:
            xtcklabs = [' - ' + str(0.9*maxth),thi,' + ' + str(0.9*maxth)];
        else:
            xtcklabs = ['','',''];
        ax = plt.axes([0.01 + i*(w+0.01), 0.08+(nrows-irow)*(h+0.01), w,h]);
        ax.set_xlim(-maxth,maxth);
        hy = max(J[i,:]) - min(J[i,:]);
        ax.set_ylim(min(J[i,:]),min(J[i,:]) + 1.2*hy);
        plt.xticks(xtx,xtcklabs);
        plt.yticks([],[]);
        ax.set_title('Cost as a function of ' + thi,pad=-16);
        ax.tick_params(which='both',direction='in'); 
        plt.plot(theta,J[i,:],color=col,linestyle=styl);


def plotP(dat,th,col,truth,nrows=1,irow=1):
    
    maxit = int(dat.it[-1]);
    xtx = [maxit/10,maxit*9/10];
    if irow == nrows:
        xtcklabs = ['%d' % xtx[0],'%d' % xtx[1]];
    else:
        xtcklabs = ['',''];
    
    xbet = [0.02, 0.02]; ybet = [0.02,0.02]
    lef = 0.025; bot = 0.08; top = 0.01;

    # Initialize parameter axes
    axp = [];
    for i in range(len(th)):
        axp.append(mySubPlot2([len(th),1],[nrows,1],[i+1,1],[1+nrows-irow,1],xbet,ybet,lef,bot,top));
        axp[i].set_xlim(1,maxit);
        axp[i].set_xticks([]);
        axp[i].set_title('Parameter ' + str(i+1) +': \u03b8$_{' + str(i+1) + '}$',pad=-16);
        axp[i].tick_params(which='both',direction='in');   
        if truth:
            plt.plot(dat.it[[0,-1]],[th[i],th[i]],color='black');
        plt.plot(dat.it,dat.th[i,:],color=col);
        
        ymin = -0.5*th[i];
        ymax =  2*th[i];
        plt.ylim([ymin,ymax]);
        plt.xticks(xtx,xtcklabs);
        if irow == nrows:
            axp[i].xaxis.set_label_coords(0.5, -0.03);
            axp[i].set_xlabel('iter.');
    return;
    
def myplot(ax,x,y,c=tudcols[0],xlab='',ylab='',tit='',scal='linlin',xtx = [],ytx = []):
    plt.sca(ax);
    plt.plot(x,y,color=c);
    ax.xaxis.set_label_coords(0.5, -0.05);
    if scal == 'linlog' or scal == 'loglog':
        ax.set_yscale('log');
    if scal == 'loglin' or scal == 'loglog':
        ax.set_xscale('log');
    ax.tick_params(which='both',direction='in');
    if not isempty(xtx):
        ax.set_xticks(xtx[0]);
        ax.set_xticklabels(xtx[1]);
    if not isempty(ytx):
        ax.set_yticks(ytx[0]);
        ax.set_yticklabels(ytx[1]);
    ax.set_title(tit,pad=-16);
    ax.set_ylabel(ylab);
    ax.set_xlabel(xlab);
    ax.set_xlim(x[[0,-1]]);
    h = max([max(y)-min(y),0.1]);
    ax.set_ylim([min(y)-0.1*h,min(y) + 1.3*h]);
def upperDiag(Min,nx):
    
    if Min == []:
        n = nx;
        N = 2;
        M = zeros((nx,nx,N));
    else: 
        try:
            n = len(Min[:,0,0]);
            N = len(Min[0,0,:]);
            M = Min;
        except: 
            n = len(Min[:,0]);
            N = 2;
            M = zeros((n,n,N));
            M[:,:,0] = Min; M[:,:,1] = Min;
        
    Mout = zeros((int(n*(n+1)/2),N));
    Mi = [];   
    for i in range(0,n):
        for j in range(0,i+1):
            Mout[i+j,:] = M[i,j,:];
            Mi.append('(' + str(i+1) + ','+ str(j+1) + ')');
                
    return Mout,Mi;

def updatecounters(cc,rc,nc):
    if cc == nc:
        rc += -1;
        cc = 1;
    else: 
        cc +=1;
    return cc, rc;      
      
def plotPEdata(dat,plcol,leg=[]):
    
    t = dat.t;
    
    # Preparation;
    xbet = [0.05, 0.025]; ybet = [0.06,0.01]
    lef = 0.025; bot = 0.06; top = 0.01;

    Q,Qi = upperDiag(dat.Q,len(dat.xc[:,0]));
    R,Ri = upperDiag(dat.R,len(dat.yp[:,0]));
    
    try:
        nth = len(dat.th[:,0]);
        nit = len(dat.th[0,:]);
        it = linspace(1,nit,nit);
    except:
        nth = len(dat.th);
        nit = 2;
        th = zeros((nth,2));
        th[:,0] = dat.th;
        th[:,1] = dat.th;
        dat.th = th;
        it = linspace(1,1e5,nit);
    
    
    nc = int(ceil((len(Q[:,0])+len(R[:,0])+\
                   nth+1)/2));
    nr1 = 2;
    nr2 = len(dat.xc[:,0])+len(dat.yp[:,0]);
    
                            

    # Cost
    if len(dat.J)!=0:
        mySubPlot2([1,nc],[2,nr1],[1,1],[2,2],xbet,ybet,lef,bot,top);
        J = dat.J[0,:]-amin(dat.J[0,:])/(amax(dat.J[0,:])-amin(dat.J[0,:]));
        plt.plot(it,J,color=plcol);
        plt.title('J',pad=-12);
        plt.xlim(it[[0,-1]]);
    
    rc = 2; cc = 2;
    # Parameters
    for i in range(0,len(dat.th[:,0])):
        mySubPlot2([1,nc],[2,nr1],[1,cc],[2,rc],xbet,ybet,lef,bot,top);
        plt.plot(it,dat.th[i,:],color=plcol);
        plt.title('p('+str(i+1)+')',pad=-12);
        if rc == 1:
            plt.xlabel('it');
        else:
            plt.xticks([],[]);
        plt.xlim(it[[0,-1]]);
        cc,rc = updatecounters(cc,rc,nc);
    
    if len(dat.Q)!=0:
        # Hyper-parameters (1/2)
        for i in range(0,len(Q[:,0])):
            mySubPlot2([1,nc],[2,nr1],[1,cc],[2,rc],xbet,ybet,lef,bot,top);
            plt.plot(it,Q[i,:],color=plcol);
            plt.title('Q' + Qi[i],pad=-12);
            if rc == 1:
                plt.xlabel('it');
            else:
                plt.xticks([],[]);
            plt.xlim(it[[0,-1]]);
            cc,rc = updatecounters(cc,rc,nc);
            
    if len(dat.R)!=0:
        # Hyper-parameters (2/2)
        for i in range(0,len(R[:,0])):
            mySubPlot2([1,nc],[2,nr1],[1,cc],[2,rc],xbet,ybet,lef,bot,top);
            plt.plot(it,R[i,:],color=plcol);
           
            plt.title('R' + Ri[i],pad=-12);
            if rc == 1:
                plt.xlabel('it');
            else:
                plt.xticks([],[]);
            plt.xlim(it[[0,-1]]);
            cc,rc = updatecounters(cc,rc,nc);
    
    
    # output
    for i in range(0,len(dat.yp[:,0])):
        mySubPlot2([1,1],[2,nr2],[1,1],[1,nr2-i],xbet,ybet,lef,bot,top);
        plt.plot(t,dat.yp[i,:],color=plcol);
        plt.title('y(' + str(i+1) + ')',pad=-12);
        plt.xticks([],[]);
        plt.xlim(t[[0,-1]]);
        
    # hidden state
    for i in range(0,len(dat.xc[:,0])):
        mySubPlot2([1,1],[2,nr2],[1,1],[1,nr2-len(dat.yp[:,0])-i],\
                   xbet,ybet,lef,bot,top);
        plt.plot(t,dat.xc[i,:],color=plcol);
        if i == len(dat.xc[:,0])-1:
            plt.xlabel('t (s)');
        else:
            plt.xticks([],[]);
        plt.title('x(' + str(i+1) + ')',pad=-12);
        plt.xlim(t[[0,-1]]);
        plt.legend(leg);

def fixdims(x,y):
    if x == [] or y == []:
        return x,y;
    else:
        try: 
            nx = len(x[:,0]); 
            N = len(x[0,:]); 
            xt = x;
        except: 
            nx = 1; 
            N = len(x);
            xt = zeros((1,N));
            xt[0,:] = x;
        try: 
            ny = len(y[:,0]); 
            yf = y;
        except: 
            ny = 1; 
            yf = zeros((ny,N));
            yf[0,:] = y;
            
        if nx == ny:
            xf = xt;
        else:
            xf = zeros((ny, N));
            for i in range(0,ny):
                xf[i,:] = xt[0,:];
    return xf,yf;
           
def plotData(x,xs,y,ys,scal,tit,xlab,ylab,cols=defcols,xlim=[], ylim=[],lg=1,\
              xtx=1,xtlbs=[],ytx=1,ytlbs=[]): 

    leg = [];
    if x != [] and y != []:
        x,y   = fixdims(x,y);
        n  = len(x[:,0]);
        for i in range(0,n):
            if scal == 'log':
                plt.xscale('log');
                plt.plot(x[0,2:-1],10*log10(y[i,2:-1]),\
                          linestyle=':', color=cols[2*i]);
            else:
                plt.plot(x[0,:],y[i,:],linestyle=':', color=cols[2*i]);
            leg.append("dim " + str(i+1) + " measured");
    
    if xs != [] and ys != []:  
        xs,ys = fixdims(xs,ys);
        ns = len(xs[:,0]);
        for i in range(0,ns):
            if scal == 'log':
                plt.plot(xs[0,:],10*log10(ys[i,:]),linestyle='-',\
                       color=cols[2*i+1]);   
            else:
                plt.plot(xs[0,:],ys[i,:],linestyle='-',color=cols[2*i+1]);         
            leg.append("dim " + str(i+1) + " baseline");
        
    if len(ylim) >= 2 and ylim[1] > ylim[0]:
        plt.ylim(ylim[0:2]);
    if len(xlim) >= 2 and xlim[1] > xlim[0]:
        plt.xlim(xlim[0:2])
    if lg==1:  
        plt.legend(leg);
    plt.title(tit,pad=-16);
    plt.xlabel(xlab);
    
    plt.ylabel(ylab);
    if xtx != 1:
        plt.xticks(xtx,xtlbs);
    if ytx != 1:
        plt.yticks(ytx,ytlbs);  
    return 0;

def mySubPlot2(ncol,nrow,icol,irow,xbet,ybet,left,bottom,top):
    
    h  =[0,0]; w = [0,0];
    
    h[0] = (1 - (nrow[0]-1)*ybet[0] - bottom - top)/nrow[0];
    w[0] = (1 - (ncol[0]  )*xbet[0] - left        )/ncol[0];
    
    h[1] = (h[0] - (nrow[1]-1)*ybet[1])/nrow[1];
    w[1] = (w[0] - (ncol[1]-1)*xbet[1])/ncol[1];
    
    ax1 = left   + (icol[0]-1)*(w[0]+xbet[0]) + (icol[1]-1)*(w[1]+xbet[1]);
    ax2 = bottom + (irow[0]-1)*(h[0]+ybet[0]) + (irow[1]-1)*(h[1]+ybet[1]);
    ax3 = w[1];
    ax4 = h[1];
        
    ax = plt.axes((ax1,ax2,ax3,ax4));
    
    return ax;

def mySubPlot(ncol,nrow,icol,irow,xbet,ybet,left,bottom,top):
    h = (1 - (nrow-1)*ybet - bottom - top)/nrow;
    w = (1 -     ncol*ybet - left        )/ncol;
    
    try: 
        nver = 1+irow[1]-irow[0];
        irow = irow[0];
    except:  
        nver = 1;
    try: 
        nhor = 1+icol[1]-icol[0];
        icol = icol[0];
    except: 
        nhor = 1;
    
    ax1 = left   + (icol-1)*(w+xbet);
    ax2 = bottom + (irow-1)*(h+ybet);
    if nhor == 1:
        ax3 = w;
    else:
        ax3 = nhor*w + (nhor-1)*xbet;
        
    if nver == 1:
        ax4 = h;
    else:
        ax4 = nver*h + (nver-1)*ybet;
        
    ax = plt.axes((ax1,ax2,ax3,ax4));
    
    return ax;

def plotSimulation(t,u,x,y,w=[],v=[],col=defcols[1],setlims=True):
    if w == []: w = zeros((2,2));
    if v == []: v = zeros((2,2));

    xbet=[0.01,0];
    ybet=[0.005,0.005];
    left=0.03;
    bottom=0.06;
    top = 0.05;
    try:
        nu = len(u[:,0]);
    except:
        nu = 0;
    try:
        nx = len(x[:,0]);
    except:
        nx = 0;
    try:
        ny = len(y[:,0]);
    except:
        ny = 0;
    nrow1 = nu + nx + ny;
    if sum(sum(w)) == 0 and sum(sum(v)) == 0:
        nrow2 = 1;
    else:
        nrow2 = nx+ny;
        nrow1 = nrow1 + 1;
      
    for i in range(0,ny):
        ax = mySubPlot2([1,1],[nrow1,1],[1,1],[nrow1-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,y[i,:],color=col);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            
            ymin = min(y[i,:]);
            h = max(y[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
        
        if nu != 0 and nx != 0 and i != ny-1:
            plt.xticks([],[]);
        if ny == 1:
            plt.title('Output: y',pad=-16);
        else:
            plt.title('Output '+ str(i+1) + ': $y_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in'); 
    
    for i in range(0,nx):
        ax = mySubPlot2([1,1],[nrow1,1],[1,1],[nrow1-ny-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,x[i,:],color=col);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            ymin = min(x[i,:]);
            h = max(x[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
        
        if nu != 0 and i != nx-1:
            plt.xticks([],[]);
        if nx == 1:
            plt.title('Hidden state: x',pad=-16);
        else:
            plt.title('Hidden state '+ str(i+1) + ': $x_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in'); 
            
    for i in range(0,nu):
        ax = mySubPlot2([1,1],[nrow1,1],[1,1],[nrow1-ny-nx-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,u[i,:],color=col);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            
            ymin = min(u[i,:]);
            h = max(u[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.3*h]);
        
        if i != nu-1:
            plt.xticks([],[]);
        if nu == 1:
            plt.title('Input: u',pad=-16);
        else:
            plt.title('Input '+ str(i+1) + ': $u_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in');  
            
    if nrow2 != 1:
        for i in range(0,ny):
            mySubPlot2([1,1],[nrow1,nrow2],[1,1],[nrow1-ny-nx-nu,nrow2-i],xbet,ybet,left,bottom,top);
            plt.plot(t,v[i,:],color=col);
            plt.xlim([0,t[-1]]);
            plt.xticks([],[]);
            if ny == 1:
                plt.ylabel("v");
            else:
                plt.ylabel("$v_" + str(i+1) +  "$");
        for i in range(0,nx):
            mySubPlot2([1,1],[nrow1,nrow2],[1,1],[nrow1-ny-nx-nu,nrow2-ny-i],xbet,ybet,left,bottom,top);
            plt.plot(t,w[i,:],color=col);
            plt.xlim([0,t[-1]]);
            plt.xticks([],[]);
            if nx == 1:
                plt.ylabel("w");
            else:
                plt.ylabel("$w_" + str(i+1) +  "$");
    
    ax.xaxis.set_label_coords(0.5, -0.05);
    plt.xlabel('t (s)')
        

def plotOPE(t,x,y,th,col=defcols[1],setlims=True,styl='-'):

    xbet=[0.01,0];
    ybet=[0.005,0.005];
    left=0.03;
    bottom=0.06;
    top = 0.05;
    
    N = len(t);

    try:
        nth = len(th[:,0]);
    except:
        if isempty(th):
            nth = 0;
        else:
            nth = len(th);
            th0 = zeros(nth)
            th0[:] = th[:];
            th = zeros((nth,N));
            for i in range(0,nth):
                th[i,:] = th0[i];        
    try:
        nx = len(x[:,0]);
    except:
        nx = 0;
    try:
        ny = len(y[:,0]);
    except:
        ny = 0;
        
    nplots = nth + nx + ny;
      
    for i in range(0,nth):
        ax = mySubPlot2([1,1],[nplots,1],[1,1],[nplots-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,th[i,:],color=col,linestyle=styl);
        plt.xticks([],[]);
        if setlims:
            plt.xlim([0,t[-1]]);
            ymin = min(th[i,:]);
            h = max(th[i,:]) - ymin;
            if h < 1e-1:
                ymin = 0;
                h = max(th[i,:]);
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
        if nth == 1:
            plt.title('Parameter: \u03b8', pad=-16);
        else:
            plt.title('Parameter ' + str(i+1) +': \u03b8$_{' + str(i+1) + '}$',pad=-16);
        ax.tick_params(which='both',direction='in');  
    
    for i in range(0,ny):
        ax = mySubPlot2([1,1],[nplots,1],[1,1],[nplots-nth-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,y[i,:],color=col,linestyle=styl);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            
            ymin = min(y[i,:]);
            h = max(y[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
        
        plt.xticks([],[]);
        if ny == 1:
            plt.title('Output: y',pad=-16);
        else:
            plt.title('Output '+ str(i+1) + ': $y_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in'); 
    
    for i in range(0,nx):
        ax = mySubPlot2([1,1],[nplots,1],[1,1],[nplots-nth-ny-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,x[i,:],color=col,linestyle=styl);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            ymin = min(x[i,:]);
            h = max(x[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
            
        if i != nx-1:
            plt.xticks([],[]);
        else:
            plt.xlabel('t (s)')
            ax.xaxis.set_label_coords(0.5, -0.05);
            
        if nx == 1:
            plt.title('Hidden state: x',pad=-16);
        else:
            plt.title('Hidden state '+ str(i+1) + ': $x_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in');     
    plt.xlabel("t (s)");
    
def plotPE(t,xe,ye,xr,yr,the,thr,maxit,col=defcols[1]):

    xbet=[0.02,0.005];
    ybet=[0.05,0.005];
    lef=0.03;
    bot=0.06;
    top = 0.05;

    try:
        nth = len(th[:,0]);
    except:
        if isempty(th):
            nth = 0;
        else:
            nth = len(th);
            th0 = zeros(nth)
            th0[:] = th[:];
            th = zeros((nth,maxit));
            for i in range(0,nth):
                th[i,:] = th0[i];        
    try:
        nx = len(x[:,0]);
    except:
        nx = 0;
    try:
        ny = len(y[:,0]);
    except:
        ny = 0;
        
    it = linspace(1,maxit,maxit);
    xtx = [0.1*maxit,0.9*maxit];
    xtcklabs = ['%d' % xtx[0],'%d' % xtx[1]];
    
    rat = 3/7;
    
    hp = rat-top-ybet[0];
    wp = (1-lef-(nth-1)*xbet[0]-xbet[1])/nth
    hs = (1-rat-bot-(nx+ny)*ybet[1])/(nx+ny);
    ws = 1-lef-xbet[1];
    
    for i in range(0,nth):
        ax = plt.axes([lef+i*(wp+xbet[0]),1-rat+ybet[0],wp,hp])
        #mySubPlot2([1,1],[nplots,1],[1,1],[nplots-i,1],xbet,ybet,left,bottom,top);
        plt.plot(it,th[i,:],color=col,linestyle=styl);
        if setlims:
            plt.xlim([1,maxit]);
            ymin = min(th[i,:]);
            h = max(th[i,:]) - ymin;
            if h < 1e-1:
                ymin = 0;
                h = max(th[i,:]);
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
        if nth == 1:
            plt.title('Parameter: \u03b8', pad=-16);
        else:
            plt.title('Parameter ' + str(i+1) +': \u03b8$_{' + str(i+1) + '}$',pad=-16);
        ax.tick_params(which='both',direction='in');  
        plt.xticks(xtx,xtcklabs)
        ax.xaxis.set_label_coords(0.5, -0.03);
        ax.set_xlabel('iter.');
    
    for i in range(0,ny):
        ax = plt.axes([lef,bot+(nx+ny-1-i)*(hs+ybet[1]),ws,hs])
        #ax = mySubPlot2([1,1],[nplots,1],[1,1],[nplots-nth-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,y[i,:],color=col,linestyle=styl);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            
            ymin = min(y[i,:]);
            h = max(y[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
        
        plt.xticks([],[]);
        if ny == 1:
            plt.title('Output: y',pad=-16);
        else:
            plt.title('Output '+ str(i+1) + ': $y_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in'); 
    
    for i in range(0,nx):
        ax = plt.axes([lef,bot+(nx-1-i)*(hs+ybet[1]),ws,hs])
        #ax = mySubPlot2([1,1],[nplots,1],[1,1],[nplots-nth-ny-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,x[i,:],color=col,linestyle=styl);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            ymin = min(x[i,:]);
            h = max(x[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.6*h]);
            
        if i != nx-1:
            plt.xticks([],[]);
        else:
            plt.xlabel('t (s)')
            ax.xaxis.set_label_coords(0.5, -0.05);
            
        if nx == 1:
            plt.title('Hidden state: x',pad=-16);
        else:
            plt.title('Hidden state '+ str(i+1) + ': $x_' + str(i+1) +  '$',pad=-16);
        ax.tick_params(which='both',direction='in');     
    plt.xlabel("t (s)");
    
    
def plotGS(t,u,col=defcols[1],setlims=True,style='-'):


    xbet=[0.01,0];
    ybet=[0.005,0.005];
    left=0.02;
    bottom=0.05;
    top = 0.01;
    
    nu = len(u[:,0]);


    for i in range(0,nu):
        ax = mySubPlot2([1,1],[nu,1],[1,1],[nu-i,1],xbet,ybet,left,bottom,top);
        plt.plot(t,u[i,:],color=col,linestyle=style);
        
        if setlims:
            plt.xlim([0,t[-1]]);
            
            ymin = min(u[i,:]);
            h = max(u[i,:]) - ymin;
            plt.ylim([ymin - 0.1*h,ymin + 1.5*h]);
        
        if i != nu-1:
            plt.xticks([],[]);
        else:
            plt.xlabel('t (s)')
            ax.xaxis.set_label_coords(0.5, -0.05);

        if i ==0:
            plt.title('Signal layer ' + str(i+1) + ': ' + r'$\phi(t)$',pad=-20);
        elif i ==1:
            #plt.title('Signal layer ' + str(i+1) + ': ' + r'$\frac{d\phi(t)}{dt}$',pad=-20);
            plt.title('Signal layer ' + str(i+1) + ': ' + r'$\dot{\phi}(t)$',pad=-20);
        elif i == 2:
            plt.title('Signal layer ' + str(i+1) + ': ' + r'$\frac{d^' + str(i) \
                      + '\phi(t)}{dt^' + str(i) + '}$',pad=-20);
            plt.title('Signal layer ' + str(i+1) + ': ' + r'$\ddot{\phi}(t)$',pad=-20);
        elif i > 2:
            plt.title('Signal layer ' + str(i+1) + ': ' + r'$\phi^{(' + str(i) + ')}(t)$',pad=-20);
        ax.tick_params(which='both',direction='in');  

def showNoiseData(N,corwin='def',twin='def'):
    
        t,w,x,xs,y,ys,f,fs,A,As,xc,xcs,yc,ycs = N.getData(corwin,twin);
        pd,k,n = N.getParams();
        
        xbet = 0.03; 
        ybet = 0.07
        lef = 0.05; 
        bot = 0.06; 
        top = 0.01;
        w1 = 0.6;
        h1 = 0.56;
        
        cols = [tudcols[0], tudcols[0],tudcols[1], tudcols[1],tudcols[2], tudcols[2]];
                
        font = {'size'   : 14}
        plt.rc('font', **font);
        plt.rc('axes', titleweight='bold');
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
           
        ymean = (log10(abs(amax(As)))+log10(abs(amin(As))))/2
        #ylim = [ymean-20,ymean+40];
        ylim = [-20,40];
        #h = ylim[1] - ylim[0];
        #ylim[1] = ylim[0] + h;
        xlim = [amin(fs), amax(fs)];
        
        
        ax1 = plt.axes((lef,1-h1-top,w1,h1));
        ax1.xaxis.set_label_coords(0.5, -0.05)
        ax1.tick_params(which='both',direction='in');
        #plt.plot(fs,10*log10(abs(As)),color='black');
        plt.plot(f[0,:],10*log10(abs(A[0,:])),color=tudcols[0],linestyle=':');
        
        plt.xscale('log');
        plt.xlabel("$\omega$ (Hz)");
        plt.ylabel("$|H^2(\omega)|$ (dB)");
        plt.title("Power Spectral Density",pad=-16);
        plt.xlim(xlim);
        plt.ylim(ylim);

        wh = 1.1*amax(abs(w));
        h = 1.1*2*n*wh;
        wlims = [0,h];
        wtx = []; wtklbs = [];
        ha = 1.1;
        alims = [-0.1,1.1*n*ha];
        
        if n==1:
            wh  = amax(w) - amin(w);
            wlims = [amin(w) - 0.1*wh, amax(w) + 0.3*wh];
            
            wtx.append(pd[0,0]);
            wtx.append(2*pd[0,1]);
            wtx.append(-2*pd[0,1]);
            wtklbs.append('%d' % int(pd[0,0]));
            wtklbs.append('%d' % int(2*pd[0,1]));
            wtklbs.append('%d' % int(-2*pd[0,1]));

            ax2 = plt.axes((lef+w1+xbet,1-top-h1,1-lef-2*xbet-w1,h1));
            ax2.tick_params(which='both',direction='in');
            ax2.xaxis.set_label_coords(0.5, -0.05)
            plotData([],-xc[0,:],[],yc[0,:]+(n-1)*ha,\
            'lin',"Autocorrelation","$\u03C4$","E[w[k]w[k+\u03C4]]",cols[2*0:2*0+2],[-k,k],\
            alims,0,1,[],[],[]);
                      
            ax3 = plt.axes((lef,bot,1-xbet-lef,1-bot-ybet-top-h1));
            ax3.xaxis.set_label_coords(0.5, -0.05)
            ax3.tick_params(which='both',direction='in');
            plotData(t,[],w[0,:]-pd[0,0],\
                     [],'lin',"Noise signal",\
                     "t (s)", "w",cols[2*0:2*0+2],t[[0,-1]],wlims,0,\
                     1,[],wtx,wtklbs);
        else:
            
            for i in range(0,n):   
                wtx.append(int((2*(n-i-1)+1)*wh));
                wtx.append(int((2*(n-i-1)+1)*wh+2*pd[i,1]));
                wtx.append(int((2*(n-i-1)+1)*wh-2*pd[i,1]));
                wtklbs.append('%d' % int(pd[i,0]));
                wtklbs.append('%d' % int(2*pd[i,1]));
                wtklbs.append('%d' % int(-2*pd[i,1]));
    
                ax2 = plt.axes((lef+w1+xbet,1-top-h1,1-lef-2*xbet-w1,h1));
                ax2.tick_params(which='both',direction='in');
                ax2.xaxis.set_label_coords(0.5, -0.05)
                plotData([],-xc[0,:],[],yc[i,:]+(n-1-i)*ha,\
                'lin',"Autocorrelation","$\u03C4$","E[w[k]w[k+\u03C4]]",cols[2*i:2*i+2],[-k,k],\
                alims,0,1,[],[],[]);
                          
                ax3 = plt.axes((lef,bot,1-xbet-lef,1-bot-ybet-top-h1));
                ax3.xaxis.set_label_coords(0.5, -0.05)
                ax3.tick_params(which='both',direction='in');
                plotData(t,[],w[i,:]-pd[i,0]+wtx[i*3],\
                         [],'lin',"Noise signal",\
                         "t (s)", "w",cols[2*i:2*i+2],t[[0,-1]],wlims,0,\
                         1,[],wtx,wtklbs);
        
    
    
    
    
    