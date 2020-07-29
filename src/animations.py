import matplotlib.pyplot as plt
from numpy import zeros, linspace
from src.visualizations import tudcols 
 
class animaxis:   
    
    def __init__(self, f, ax, tit='def',\
                            xlim='def',\
                            ylim='def',\
                            xcoords='def',\
                            xtx='def',\
                            xtlbs='def',\
                            ytx='def',\
                            ytlbs='def',
                            xlab='def',
                            ylab='def',
                            nlines=1,
                            pad=-16):
        
        plt.sca(ax);
        
        if tit=='def':
            self.tit = ax.get_title();
        else:
            self.tit = tit;
            
        if xlim=='def':
            self.xlim = ax.get_xlim();
        else:
            self.xlim = xlim;
            
        if ylim=='def':
            self.ylim = ax.get_ylim();
        else:
            self.ylim = ylim;
            
        if xcoords=='def':
            self.xcoords = ax.xaxis.get_label_coords();
        else:
            self.xcoords = xcoords;
          
        # xticks + labels
        if xtx!='def' and xtlbs!='def':
            plt.xticks(xtx,xtlbs);

        # yticks + labels
        if ytx!='def' and ytlbs!='def':
            plt.yticks(ytx,ytlbs);
           
        # Axis labels
        if xlab=='def':
            self.xlab = ax.get_xlabel();
        else:
            self.xlab = xlab;
        if ylab=='def':
            self.ylab = ax.get_ylabel();
        else:
            self.ylab = ylab;
        
        # Set parameters to axes
        ax.set_xlabel(self.xlab);
        ax.set_ylabel(self.ylab);
        ax.set_xlim(self.xlim);
        ax.set_ylim(self.ylim);
        ax.set_title(self.tit, pad=pad);
        ax.xaxis.set_label_coords(xcoords[0],xcoords[1]);
        
        self.ax = ax;
        self.f = f;
        self.ax.tick_params(which='both',direction='in');
        self.bgs = self.f.canvas.copy_from_bbox(ax.bbox);
        self.lines = [];
        
    
    def add_line(self,xdat=[],ydat=[],col='black',wdth=1,style='-'):
        self.lines.append(plt.plot(xdat,ydat,linewidth=wdth,color=col,linestyle=style)[0]);
        
    def update_lines(self,xdata,ydata,i='nan'):
        self.f.canvas.restore_region(self.bgs);
        if i=='nan':
            for i in range(len(self.lines)):
                self.lines[i].set_data(xdata, ydata);
                self.ax.draw_artist(self.lines[i]);
        else:
             self.lines[i].set_data(xdata, ydata);
             self.ax.draw_artist(self.lines[i]);
        self.ax.figure.canvas.blit(self.ax.bbox);

class PEanimation:
    
    def __init__(self,f,t,x,y,thr,maxit,leg=[],col=tudcols[0],save=False,dpi=100,fmt='png',plotstates=True,n=1,styles=['-','--','-.',':']):
        
        nth = len(thr)
        nx  = len(x[:,0]);
        ny  = len(y[:,0]);
        self.nth = nth;
        self.nx = nx;
        self.ny = ny;
        self.t = t;
        self.fmt = fmt;
        self.dpi=dpi;
        self.n = n;
        self.pls = plotstates;
        
        th = zeros((nth,2));
        th[:,0] = thr;
        th[:,1] = thr;
            
        xbet=[0.02,0.01];
        ybet=[0.05,0.005];
        lef=0.02;
        bot=0.06;
        top = 0.05;

        self.it = linspace(1,maxit,maxit);
        
        xtcks = [0.1*maxit,0.9*maxit];
        xtcklabs = ['%d' % xtcks[0],'%d' % xtcks[1]];
        rat = 3/7;
        if plotstates:
            hp = rat-top-ybet[0];
            hs = (1-rat-bot-(nx+ny)*ybet[1])/(nx+ny);
            ws = 1-lef-xbet[1];
            yp = 1-rat+ybet[0];
        else:
            hp = 1-top-bot;
            yp = bot
        wp = (1-lef-(nth-1)*xbet[0]-xbet[1])/nth
    
        self.axth = [];
        for i in range(0,nth):
            ax = plt.axes([lef+i*(wp+xbet[0]),yp,wp,hp]);
            xlimit=[1,maxit];
            ymin = 0;
            h = max(th[i,:]);
            ylimit = [ymin - 0.1*h,ymin + 1.6*h]; 
            ytcks =   [0,th[i,0]];
            ytcklabs = ['%d ' % 0,'%d' % th[i,0]];
            if nth == 1:
                title = 'Parameter: \u03b8';
            else:
                title = 'Parameter ' + str(i+1) +': \u03b8$_{' + str(i+1) + '}$';
            self.axth.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xtx=xtcks,\
                                xtlbs=xtcklabs,\
                                ytx=ytcks,\
                                ytlbs=ytcklabs,\
                                xlab='iter.'));
            self.axth[i].add_line(self.it[[0,-1]],th[i,:],col='black',wdth=1);
            for j in range(0,self.n):
                self.axth[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
        
        plt.legend(leg,loc=4);
        if plotstates:
            self.axx = [];
            for i in range(0,nx):
                ax = plt.axes([lef,bot+(nx-1-i)*(hs+ybet[1]),ws,hs]);
                xlimit=t[[0,-1]];
                ymin = min(x[i,:]);
                h = max(x[i,:]) - ymin;
                ylimit = [ymin - 0.1*h,ymin + 1.6*h];
                if nx == 1:
                    title = 'Hidden state: x';
                else:
                    title = 'Hidden state ' + str(i+1) +': x$_{' + str(i+1) + '}$';
                if i==nx-1:
                    xlbl  ='t (s)';
                    xtcks = 'def';
                    xtcklabs = 'def';
                else:
                    xlbl = '';
                    xtcks = [];
                    xtcklabs = '';
                    
                yrng = max(abs(min(x[i,:])),abs(min(x[i,:]))); 
                ytcks =   [2*int(-0.5*yrng),0,2*int(0.5*yrng)];
                ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
                    
                self.axx.append(animaxis(f, ax, tit=title,\
                                    xlim=xlimit,\
                                    ylim=ylimit,\
                                    xcoords=(0.5, -0.03),\
                                    xlab = xlbl,\
                                    xtx = xtcks,\
                                    xtlbs = xtcklabs,\
                                    ytx = ytcks,\
                                    ytlbs = ytcklabs));
                self.axx[i].add_line(t,x[i,:],col='black',wdth=1);
                for j in range(0,self.n):
                    self.axx[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
                    
                
            self.axy = [];
            for i in range(0,ny):
                ax = plt.axes([lef,bot+(nx+ny-1-i)*(hs+ybet[1]),ws,hs]);
                xlimit=t[[0,-1]];
                ymin = min(y[i,:]);
                h = max(y[i,:]) - ymin;
                ylimit = [ymin - 0.1*h,ymin + 1.6*h];
                if ny == 1:
                    title = 'Output: y';
                else:
                    title = 'Output ' + str(i+1) +': y$_{' + str(i+1) + '}$';
                
                ytcks =   [2*int(0.5*min(y[i,:])),0,2*int(0.5*max(y[i,:]))];
                ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
                    
                self.axy.append(animaxis(f, ax, tit=title,\
                                    xlim=xlimit,\
                                    ylim=ylimit,\
                                    xcoords=(0.5, -0.03),\
                                    xtx = [],\
                                    xtlbs = '',\
                                    ytx = ytcks,\
                                    ytlbs = ytcklabs));
                self.axy[i].add_line(t,y[i,:],col='black',wdth=1);
                for j in range(0,self.n):
                    self.axy[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
            
        plt.show(False);        
        plt.draw();
        plt.pause(5e-1);
        plt.savefig('frames/frame%d.'% 0 , dpi=self.dpi);
        
    def updateAnim(self,k,xe,ye,the):
        
        for i in range(0,self.nth):
            for j in range(0,self.n):
                self.axth[i].update_lines(self.it[:k],the[j][i,:k],1+j);
        if self.pls:
            for i in range(0,self.nx):
                for j in range(0,self.n):
                    self.axx[i].update_lines(self.t,xe[j][i,:],1+j);
            
            for i in range(0,self.ny):
                for j in range(0,self.n):
                    self.axy[i].update_lines(self.t,ye[j][i,:],1+j);
        
        plt.pause(1e-2);
        plt.savefig('frames/frame%d.'% k + self.fmt , dpi=self.dpi);
        
class OPEanimation:
    
    def __init__(self,f,t,x,y,thr,leg=[],col=tudcols[0],save=False,win='def',dpi=100,fmt='png'):
        
        nth = len(thr)
        nx  = len(x[:,0]);
        ny  = len(y[:,0]);
        self.nth = nth;
        self.nx = nx;
        self.ny = ny;
        self.t = t;
        self.win = win;
        self.dpi=dpi;
        self.fmt = fmt;
        
        
        th = zeros((nth,2));
        th[:,0] = thr;
        th[:,1] = thr;
            
        xbet=[0.02,0.01];
        ybet=[0.06,0.005];
        lef=0.02;
        bot=0.06;
        
        hs = (1-bot-(nx+ny+nth-1)*ybet[1] - ybet[0])/(nx+ny+nth);
        ws = 1-lef-xbet[1];
    
        self.axth = [];
        for i in range(0,nth):
            ax = plt.axes([lef,1-(i+1)*(hs+ybet[1]),ws,hs]);
            ymin = 0;
            h = max(th[i,:]);
            ylimit = [ymin - 0.1*h,ymin + 1.6*h]; 
            ytcks =   [0,th[i,0]];
            ytcklabs = ['%d ' % 0,'%d' % th[i,0]];
            if nth == 1:
                title = 'Parameter: \u03b8';
            else:
                title = 'Parameter ' + str(i+1) +': \u03b8$_{' + str(i+1) + '}$';
            if i==nth-1:
                xlbl  ='t (s)';
                xtcks = 'def';
                xtcklabs = 'def';
            else:
                xlbl = '';
                xtcks = [];
                xtcklabs = '';
                
            if self.win != 'def':
                xlimit = t[[0,win]]; 
            else:
                xlimit = t[[0,-1]];
                
            self.axth.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xtx = xtcks,\
                                xtlbs = xtcklabs,\
                                xlab = xlbl,\
                                ytx=ytcks,\
                                ytlbs=ytcklabs));
            self.axth[i].add_line(self.t[[0,-1]],th[i,:],col='black',wdth=1);
            self.axth[i].add_line([],[],col=tudcols[0],wdth=1);
        
        plt.legend(leg,loc=4);
        
        self.axx = [];
        for i in range(0,nx):
            ax = plt.axes([lef,1-(i+nth+ny+1)*(hs+ybet[1])-ybet[0]+ybet[1],ws,hs]);
            ymin = min(x[i,:]);
            h = max(x[i,:]) - ymin;
            ylimit = [ymin - 0.1*h,ymin + 1.6*h];
            if nx == 1:
                title = 'Hidden state: x';
            else:
                title = 'Hidden state ' + str(i+1) +': x$_{' + str(i+1) + '}$';
            if i==nx-1:
                xlbl  ='t (s)';
                xtcks = 'def';
                xtcklabs = 'def';
            else:
                xlbl = '';
                xtcks = [];
                xtcklabs = '';
                
            if self.win != 'def':
                xlimit = t[[0,win]]; 
            else:
                xlimit = t[[0,-1]];
                
            yrng = max(abs(min(x[i,:])),abs(min(x[i,:]))); 
            ytcks =   [2*int(-0.5*yrng),0,2*int(0.5*yrng)];
            ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
                
            self.axx.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xlab = xlbl,\
                                xtx = xtcks,\
                                xtlbs = xtcklabs,\
                                ytx = ytcks,\
                                ytlbs = ytcklabs));
            self.axx[i].add_line(t,x[i,:],col='black',wdth=1);
            self.axx[i].add_line([],[],col=tudcols[0],wdth=1);
                     
        self.axy = [];
        for i in range(0,ny):
            ax = plt.axes([lef,1-(i+nth+1)*(hs+ybet[1])-ybet[0]+ybet[1],ws,hs]);
            ymin = min(y[i,:]);
            h = max(y[i,:]) - ymin;
            ylimit = [ymin - 0.1*h,ymin + 1.6*h];
            if ny == 1:
                title = 'Output: y';
            else:
                title = 'Output ' + str(i+1) +': y$_{' + str(i+1) + '}$';
            
            ytcks =   [2*int(0.5*min(y[i,:])),0,2*int(0.5*max(y[i,:]))];
            ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
            
            if self.win != 'def':
                xlimit = t[[0,win]]; 
            else:
                xlimit = t[[0,-1]];
                
            self.axy.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xtx = [],\
                                xtlbs = '',\
                                ytx = ytcks,\
                                ytlbs = ytcklabs));
            self.axy[i].add_line(t,y[i,:],col='black',wdth=1);
            self.axy[i].add_line([],[],col=tudcols[0],wdth=1);
            
        plt.show(False);        
        plt.draw();
        plt.pause(1e-1);
        plt.savefig('frames/frame%d.' % 0 + self.fmt , dpi=self.dpi);
        
    def updateAnim(self,k,xe,ye,the,frnum='def'):
        if frnum=='def':
            frnum=k;
        if self.win == 'def' or k <= self.win:
            if len(self.t) != len(xe[0,:]):
                t = self.t[:len(xe[0,:])];
            
            for i in range(0,self.nth):
                self.axth[i].update_lines(t,the[i,:],1);
                
            for i in range(0,self.nx):
                self.axx[i].update_lines(t,xe[i,:],1);
            
            for i in range(0,self.ny):
                self.axy[i].update_lines(t,ye[i,:],1);
        
        else:
            t = self.t[k-self.win:k];
            for i in range(0,self.nth):
                self.axth[i].update_lines(t,the[i,-1-self.win:-1],1);
                self.axth[i].ax.set_xlim(t[[0,-1]]);
                
            for i in range(0,self.nx):
                self.axx[i].update_lines(t,xe[i,-1-self.win:-1],1);
                self.axx[i].ax.set_xlim(t[[0,-1]]);
            
            for i in range(0,self.ny):
                self.axy[i].update_lines(t,ye[i,-1-self.win:-1],1);
                self.axy[i].ax.set_xlim(t[[0,-1]]);
        
        plt.pause(1e-2);
        plt.savefig('frames/frame%d.' % frnum + self.fmt , dpi=self.dpi);
        
        
class Filtanimation:
    
    def __init__(self,f,t,x,y,leg=[],col=tudcols[0],save=False,win='def',dpi=100,fmt='png',n=1,styles=['-','--','-.',':']):
        
        nx  = len(x[:,0]);
        ny  = len(y[:,0]);
        self.nx = nx;
        self.ny = ny;
        self.t = t;
        self.win = win;
        self.dpi=dpi;
        self.fmt = fmt;
        self.n = n;
        self.styles = styles;
        
            
        xbet=[0.02,0.01];
        ybet=[0.06,0.005];
        lef=0.02;
        bot=0.06;
        
        hs = (1-bot-(nx+ny-1)*ybet[1] - ybet[0])/(nx+ny);
        ws = 1-lef-xbet[1];
    
        self.axy = [];
        for i in range(0,ny):
            ax = plt.axes([lef,1-(i+1)*(hs+ybet[1])-ybet[0]+ybet[1],ws,hs]);
            ymin = min(y[i,:]);
            h = max(y[i,:]) - ymin;
            ylimit = [ymin - 0.1*h,ymin + 1.6*h];
            if ny == 1:
                title = 'Output: y';
            else:
                title = 'Output ' + str(i+1) +': y$_{' + str(i+1) + '}$';
            
            ytcks =   [2*int(0.5*min(y[i,:])),0,2*int(0.5*max(y[i,:]))];
            ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
            
            if self.win != 'def':
                xlimit = t[[0,win]]; 
            else:
                xlimit = t[[0,-1]];
                
            self.axy.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xtx = [],\
                                xtlbs = '',\
                                ytx = ytcks,\
                                ytlbs = ytcklabs));
            self.axy[i].add_line(t,y[i,:],col='black',wdth=1);
            for j in range(0,n):
                self.axy[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
    
        self.axx = [];
        for i in range(0,nx):
            ax = plt.axes([lef,1-(i+ny+1)*(hs+ybet[1])-ybet[0]+ybet[1],ws,hs]);
            ymin = min(x[i,:]);
            h = max(x[i,:]) - ymin;
            ylimit = [ymin - 0.1*h,ymin + 1.6*h];
            if nx == 1:
                title = 'Hidden state: x';
            else:
                title = 'Hidden state ' + str(i+1) +': x$_{' + str(i+1) + '}$';
            if i==nx-1:
                xlbl  ='t (s)';
                xtcks = 'def';
                xtcklabs = 'def';
            else:
                xlbl = '';
                xtcks = [];
                xtcklabs = '';
                
            if self.win != 'def':
                xlimit = t[[0,win]]; 
            else:
                xlimit = t[[0,-1]];
                
            yrng = max(abs(min(x[i,:])),abs(min(x[i,:]))); 
            ytcks =   [2*int(-0.5*yrng),0,2*int(0.5*yrng)];
            ytcklabs = ['%d ' % ytcks[0],'0','%d' % ytcks[2]]; 
                
            self.axx.append(animaxis(f, ax, tit=title,\
                                xlim=xlimit,\
                                ylim=ylimit,\
                                xcoords=(0.5, -0.03),\
                                xlab = xlbl,\
                                xtx = xtcks,\
                                xtlbs = xtcklabs,\
                                ytx = ytcks,\
                                ytlbs = ytcklabs));
            self.axx[i].add_line(t,x[i,:],col='black',wdth=1);
            for j in range(0,n):
                self.axx[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
            if i==0:
                plt.legend(leg,loc=4);
            
        plt.show(False);        
        plt.draw();
        plt.pause(1e-1);
        plt.savefig('frames/frame%d.' % 0 + self.fmt , dpi=self.dpi);
        
    def updateAnim(self,k,xe,ye,frnum='def'):
        if len(self.t) != len(xe[0][0,:]):
            t = self.t[:len(xe[0][0,:])];
                
        for i in range(0,self.nx):
            for j in range(0,self.n):
                self.axx[i].update_lines(t,xe[j][i,:],1+j);
        
        for i in range(0,self.ny):
            for j in range(0,self.n):
                self.axy[i].update_lines(t,ye[j][i,:],1+j);
        
        
        plt.pause(1e-2);
        plt.savefig('frames/frame%d.' % frnum + self.fmt , dpi=self.dpi);    
    
class Noiseanimation:
    
    def __init__(self,f,t,w,tit='def',col=tudcols[0],save=False,Tvid='def',fps = 25,dpi=100,fmt='png'):
        
        
        nw  = len(w[:,0]);
        Nw   = len(w[0,:]);
        if Tvid == 'def':
            Tvid = t[-1];
        else:
            Tvid = Tvid;
        Nf = int(Tvid*fps);
        
        ybet=0.02
        rght=0.06
        lef=0.02;
        bot=0.06;
        
        hn = (1-bot)/nw-ybet;
        wn = 1-lef-rght;
    
        self.t = t;
        
    
        self.axw = [];
        for i in range(0,nw):
            ax = plt.axes([lef,1-(i+1)*(hn+ybet),wn,hn]);
            ymin = min(w[i,:]);
            ymax = max(w[i,:]);
            h = ymax-ymin
            ylimit = [ymin - 0.1*h,ymax + 0.3*h]; 
            ytcks =   [];
            ytcklabs = [];
            
            if tit == 'def':
                title = 'Noise %d' % (i+1);
            else:
                title = tit[i]
            if i==nw-1:
                xlbl  ='t (s)';
                xtcks = 'def';
                xtcklabs = 'def';
            else:
                xlbl = '';
                xtcks = [];
                xtcklabs = '';
                
            self.axw.append(animaxis(f, ax, tit=title,\
                                xlim=t[[0,-1]],\
                                ylim=ylimit,\
                                xcoords=(0.6, -0.03),\
                                xtx = xtcks,\
                                xtlbs = xtcklabs,\
                                xlab = xlbl,\
                                ytx=ytcks,\
                                ytlbs=ytcklabs));
                                     
            self.axw[i].add_line(self.t[[0,-1]],[0,0],col='black',wdth=2);
            self.axw[i].add_line([],[],col=tudcols[0],wdth=1);

        plt.show(False);        
        plt.draw();
        plt.pause(1e-1);
        plt.savefig('frames/frame0.png', dpi=300);
        
        for fr in range(0,Nf):
            k = int(Nw*fr/Nf);
            for i in range(0,nw):
                self.axw[i].update_lines(t[:k],w[i,:k],1);
            
            plt.pause(1e-2);
            if save:
                plt.savefig('frames/frame%d.' % fr + fmt, dpi=dpi);

class GSanimation:
    
        def __init__(self,f,t,truth,data,col=tudcols[0],leg='def',\
                     save=False,Tvid='def',fps = 25,dpi=100,fmt='png',styles=['-','--','-.',':']):
    
            ybet  = 0.01;
            lef   = 0.01;
            rght = 0.01;
            bot   = 0.05;
            
            p = len(truth[:,0]);
            N = len(truth[0,:]);
            n = len(data);
            hf = (1-p*ybet-bot)/p;
            wf = 1-lef-rght;
            
            # Set titles
            tit=['Signal layer ' + str(1) + ': ' + r'$\phi(t)$'];
            for i in range(0,p):
                if i ==1:
                    tit.append('Signal layer 2: '+ r'$\dot{\phi}(t)$');
                elif i == 2:
                    tit.append('Signal layer 3: '+ r'$\ddot{\phi}(t)$');
                elif i > 2:
                    tit.append('Signal layer ' + str(i+1) + ': ' + \
                               r'$\phi^{(' + str(i) + ')}(t)$');
                               
            if Tvid == 'def':
                Tvid = t[-1];
            else:
                Tvid = Tvid;
            Nf = int(Tvid*fps);
        
            self.t = t;
        
    
            self.ax = [];
            for i in range(0,p):
                ax = plt.axes([lef,1-(i+1)*(hf+ybet),wf,hf]);
                ymin = min(truth[i,:]);
                ymax = max(truth[i,:]);
                h = ymax-ymin
                ylimit = [ymin - 0.1*h,ymax + 0.3*h]; 
                ytcks =   [];
                ytcklabs = [];
                if i==p-1:
                    xlbl  ='t (s)';
                    xtcks = 'def';
                    xtcklabs = 'def';
                else:
                    xlbl = '';
                    xtcks = [];
                    xtcklabs = '';
                    
                self.ax.append(animaxis(f, ax, tit=tit[i],\
                                    xlim=t[[0,-1]],\
                                    ylim=ylimit,\
                                    xcoords=(0.5, -0.03),\
                                    xtx = xtcks,\
                                    xtlbs = xtcklabs,\
                                    xlab = xlbl,\
                                    ytx=ytcks,\
                                    ytlbs=ytcklabs,\
                                    pad = -20));
                                         
                self.ax[i].add_line(self.t,truth[i,:],col='black',wdth=2);
                for j in range(0,n):
                    self.ax[i].add_line([],[],col=tudcols[0],wdth=1,style=styles[j]);
            
            if leg!='def':
                plt.legend(leg,loc=2);
    
            plt.show(False);        
            plt.draw();
            plt.pause(1e-1);
            plt.savefig('frames/frame0.png', dpi=300);
            
            for fr in range(0,Nf):
                k = int(N*fr/Nf);
                for i in range(0,p):
                    for j in range(0,n):
                        self.ax[i].update_lines(t[:k],data[j][i,:k],1+j);
                
                plt.pause(1e-2);
                if save:
                    plt.savefig('frames/frame%d.' % fr + fmt, dpi=dpi);
        
