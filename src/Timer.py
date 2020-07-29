from time import time
from numpy import floor
import sys


class Timer:
    
    __slots__ = ["_name","_i","_s","_c","_N","_l","_res","_looptime","_tic",];
    
    def __init__(self,name="Nameless",s=0,N=1,res=100):
        self.reset(name,s,N,res);
        return;
        
    def tic(self):
        self._tic =  time();
        return;
        
    def toc(self):
        _toc =  time();
        return _toc-self._tic;
        
    def update(self):
        self._looptime.append(self.toc());
        return;
    
    def reset(self, name=[], s=[], N=[], res=[]):
        
        self._looptime = 0;
        self.tic();
        if name!=[]: self._name = name;
        if s!=[]:    self._s = s;
        if N!=[]:    self._N = N;
        if res!=[]:  self._res = res;
        if self._res > self._N:
            self._res = self._N;
        self._c = 0;
        self._l = 0;
        return;    
        
    def looptime(self,i,ident=0):
        
        if (i-self._s)/self._N >= self._c/self._res \
            or self.toc()-self._looptime >= 1:
            self._c += 1;
            self._looptime = self.toc()
            t_passed = self._looptime;
            t_tot_est = self._N*t_passed/(1+i-self._s);
            t_left = t_tot_est-t_passed;
            t = [0,0,0,0,0,0,0,0,0];
            t[0] = int(floor(t_left/3600));
            t[1] = int(floor(t_left/60-60*t[0]));
            t[2] = int(round(t_left-3600*t[0]-60*t[1]));
            t[3] = int(floor(t_passed/3600));
            t[4] = int(floor(t_passed/60-60*t[3]));
            t[5] = int(round(t_passed-3600*t[3]-60*t[4]));
            t[6] = int(floor(t_tot_est/3600));
            t[7] = int(floor(t_tot_est/60-60*t[6]));
            t[8] = int(round(t_tot_est-3600*t[6]-60*t[7]));
            for i in range(0,9):
                tim = str(t[i]);
                while len(tim) < 2:
                     tim = '0' + tim;
                t[i] = tim;
            
            if ident == 1:
                txt = "";    
                txt += self._name +'\n';
                txt +='progress:              %.1f %%\n' % (100.0*(i-self._s)/self._N);
                txt +='estimated total time: ' + t[6] + ":" + t[7] + ":" + t[8] +'\n';
                txt +='time elapsed:         ' + t[3] + ":" + t[4] + ":" + t[5] +'\n';
                txt +='estimated time left:  ' + t[0] + ":" + t[1] + ":" + t[2] +'\n'; 
                self._l = len(txt);
    
                #get_ipython().magic('clear');   
                print(txt,flush=True);
            else:   
                txt = '\r' + self._name +'. ';
                txt +='progress: %.1f %%. ' % (100.0*self._c/self._res);
                txt +='Time left: ' + t[0] + ":" + t[1] + ":" + t[2] + ". "; 
                txt +='Time elapsed: ' + t[3] + ":" + t[4] + ":" + t[5] + ". ";
                sys.stdout.write(txt);
            if self._c == self._res:
                sys.stdout.write('\n');
        else:
            pass;
        return;