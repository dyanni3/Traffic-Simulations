import numpy as np
from matplotlib import pyplot as plt
#import cv2
import random
from matplotlib import animation
from matplotlib.animation import FuncAnimation
#%matplotlib inline
from scipy.optimize import curve_fit
import matplotlib.pylab as pylab
L=1000

def pb(x0,L):
	return((x0+L)%L)


class ant():
    
    def __init__(self,ids,y=0,flowc=0,stoppedc=0,cari=0):
        self.x=np.random.uniform(0,L)
        self.y=y
        self.xunp=self.x
        self.v=np.random.uniform(0,1)
        self.dt=.05
        self.tau=.1
        self.v_pref=29#+min(np.random.normal(0,6.7),6.7)
        self.R=29*2
        self.r=4.5
        self.ids=ids
        self.stopped=0
        #print(cari)
        if (cari>=flowc):   
            #print(cari)
            self.v_pref=0  
            self.stopped=1
                #print(stoppedc)

        self.ep=((self.v_pref)/self.tau)*(1-(self.r/self.R))**(-1.5)        
    def step(self,d=None,vbar=0,counter=0):
        if d==None:
            self.xunp=self.xunp+self.dt*self.v
            self.x=pb(self.x+self.dt*self.v,L)
            return('same')
        else:
            if d<=0:
                f=0
            else:
                if d<self.R:
                    #ep=((self.v_pref)/self.tau)*(1-(self.r/self.R))**(-1.5)
                    f=self.force(d)
                else:
                    f=0
        x0=self.x                
        self.v=max(0,self.v+self.dt*((self.v_pref -self.v)/self.tau -f))
        counter+=1 
        vbar=((counter-1)*vbar+self.v)/counter
        self.xunp=self.xunp+self.dt*self.v
        self.x=pb(self.x+self.dt*self.v,L)
        return(vbar,counter)
    def force(self,x):
        f=self.ep*(1-(x/self.R))**(1.5)
        if type(f)!=complex:
            return(f)
        else:
            return(0)

        
class ants():
    def __init__(self,num_ants=20,lanes=3,L=1000,flowc=0,stoppedc=0,politeness=1):
        self.lanes=lanes
        self.num_ants=num_ants
        #self.stopped=stoppedc
        self.members=[[] for i in range(lanes)]
        self.state=[[] for i in range(lanes)]
        self.vel=[[] for i in range(lanes)]
        self.ids=[[] for i in range(lanes)]
        self.xunp=[[] for i in range(lanes)]
        self.stopped=[[] for i in range(lanes)]
        self.L=L
        self.p=politeness
        cari=0
        #meanvel=0
        for l in range(lanes):
            self.members[l]=[ant(ids=i,y=l,flowc=flowc,stoppedc=stoppedc,cari=i) for i in range(num_ants)]
            self.state[l]=np.array([bug.x for bug in self.members[l]])
            self.xunp[l]=np.array([bug.xunp for bug in self.members[l]])
            self.vel[l]=np.array([bug.v for bug in self.members[l]])
            self.stopped[l]=np.array([bug.stopped for bug in self.members[l]])
            self.ids[l]=np.array([i for i in range(num_ants)])
            self.members[l]=sorted(self.members[l], key=lambda bug: bug.x)
            self.counter=0
            self.vbar=0
            
            
    def step(self):
        meanvel=0
        to_switch=[[] for i in range(self.lanes)]
        for l in range(self.lanes):
        #l=np.random.choice([0,1,2])
        #a1=np.arange(len(self.members[l]))
        #random.shuffle(a1)
        
            for i in range(len(self.members[l])):
                bug=self.members[l][i]
                switchQ=False
                if bug.y==0:
                    if random.random()>.5:
                        switchQ=self.change_lanesQ(0,1,bug,i)
                    if switchQ:
                        to_switch[1].append(bug)
                elif bug.y==1:
                    lane_to_try=np.random.choice([0,2])
                    switchQ=self.change_lanesQ(1,lane_to_try,bug,i)
                    #switchQ=self.change_lanesQ(1,0,bug,a1[i])
                    if(switchQ):
                        to_switch[lane_to_try].append(bug)
                else:
                    if random.random()>.5:
                        switchQ=self.change_lanesQ(2,1,bug,i)
                    if switchQ:
                        to_switch[1].append(bug)
            
                x_b=pb(bug.x,self.L)
                x_ol=pb(self.members[l][pb(i+1,len(self.members[l]))].x,self.L)
                dx_b=min(abs(x_ol-x_b),self.L-abs(x_ol-x_b))                
                self.vbar,self.counter=bug.step(dx_b,self.vbar,self.counter)
                               
            
        for l in range(self.lanes):
            for bug in to_switch[l]:
                self.members[l].append(bug)
                self.members[bug.y].remove(bug)
                bug.y=l
        for l in range(self.lanes):
                
            self.members[l]=sorted(self.members[l],key=lambda y: y.x)
            self.state[l]=np.array([bug.x for bug in self.members[l]])
            self.xunp[l]=np.array([bug.xunp for bug in self.members[l]])
            self.vel[l]=np.array([bug.v for bug in self.members[l]])
            self.ids[l]=np.array([bug.ids for bug in self.members[l]])
            self.stopped[l]=np.array([bug.stopped for bug in self.members[l]])
            meanvel=meanvel+np.mean(self.vel[l])
            
        meanvel=meanvel/3
 
        return(meanvel)
												
    def change_lanesQ(self,current_lane,new_lane,bug,i):
        nf_index=self.find_neighbor(bug,new_lane) 
        x_b=pb(bug.x,self.L)
        x_nl=pb(self.members[new_lane][pb(nf_index+1,len(self.members[new_lane]))].x,self.L)
        x_nf=pb(self.members[new_lane][nf_index].x,self.L)
        x_ol=pb(self.members[current_lane][pb(i+1,len(self.members[current_lane]))].x,self.L)
        x_of=pb(self.members[current_lane][pb(i-1,len(self.members[current_lane]))].x,self.L)

        #2) get relevant dx values
        #    a) get the current values
        dx_nf=min(abs(x_nl-x_nf),self.L-abs(x_nl-x_nf))
        dx_b=min(abs(x_ol-x_b),self.L-abs(x_ol-x_b))
        dx_of=min(abs(x_b-x_of),self.L-abs(x_b-x_of))
        #    b) get the values after the lane switch
        dx_bp=min(abs(x_nl-x_b),self.L-abs(x_nl-x_b))
        dx_nfp=min(abs(x_b-x_nf),self.L-abs(x_b-x_nf))
        dx_ofp=min(abs(x_ol-x_of),self.L-abs(x_ol-x_of))
        
        #3) get relevant forces
        f_current=[bug.force(dx_b)]
        f_current.append(self.members[current_lane][pb(i-1,len(self.members[current_lane]))].force(dx_of))
        f_current.append(self.members[new_lane][nf_index].force(dx_nf))
        
        f_new=[bug.force(dx_bp)]
        f_new.append(self.members[new_lane][nf_index].force(dx_nfp))
        f_new.append(self.members[current_lane][pb(i-1,len(self.members[current_lane]))].force(dx_ofp))
        if ((f_current[0]+self.p*(f_current[1]+f_current[2])>1*(f_new[0]+self.p*(f_new[1]+f_new[2]))) and \
            (bug.stopped==0)):
            return(True)
        return(False)
								
    def view(self,t):
        fig=plt.figure()
        plt.xlim(-1,self.L+1)
        plt.ylim(-1,2)
        line1, =plt.plot(np.arange(self.L),np.zeros(self.L),'k')
        line2, =plt.plot([b.x for b in self.members[0]],np.zeros(len(self.members[0])),'ro')
        line3, =plt.plot(np.arange(self.L),np.ones(self.L),'k')
        line4, =plt.plot([b.x for b in self.members[1]],np.ones(len(self.members[1])),'bo')
        line5, =plt.plot(np.arange(self.L),.5*np.ones(self.L),'k')
        line6, =plt.plot([b.x for b in self.members[2]],.5*np.ones(len(self.members[2])),'go')
        fig.savefig('./images/graph'+str(t)+'.png') #save figures
        plt.close(fig)

    def find_neighbor(self,bug,new_lane):
        x=0;i=0
        while(x<bug.x):
            try:
                x=self.members[new_lane][i].x
            except IndexError:
                return(i-1)
            i+=1
        return(i-2)

 #%%
def run(my_ants):
	a1=0
	mv=[]
#posn=np.zeros((1,3))
#dens2=np.zeros(10000)
#phi2=np.zeros(10000)
	for t in range(10000):
		meanv=my_ants.step()
#		if (t>9900):
#			my_ants.view(t)
#	a1=np.c_[my_ants.ids,my_ants.ids*0+t,my_ants.xunp]
#	posn=(np.r_[posn,a1])
		if t>9500:
			mv.append(meanv)

#		w=np.where((my_ants.state>500) &(my_ants.state<1000))[0]
#		dens2[t]=len(w)/500*1000
#		if t%1000==0:
#			print(t)
#return (posn)
	return mv

#%%
def run_sim(num_stopped_cars,pnum):
    L=1000
    rho_a=np.linspace(0.002,.15,100)
    phi_a=np.zeros([100,10])
    vel_a=np.zeros([100,10])
    #vb2=np.zeros([15,10000])
    for j in range(10):
        for i,dens in enumerate(rho_a):
            print(j)
            mv=[]
            flowc=int(dens*L)
            stoppedc=num_stopped_cars
            num_ants=stoppedc+flowc    
            my_ants=ants(num_ants=num_ants,politeness=1,flowc=flowc,stoppedc=stoppedc)
            #	posn=np.zeros(num_ants)
            mv=run(my_ants)
            phi_a[i,j]=dens*mv[-1]
            vel_a[i,j]=mv[-1]      


    rho_vel_phi_stop30car_p1=np.c_[rho_a,vel_a,phi_a]
    np.savetxt('rho_vel_phi_stop'+'%s'%str(num_stopped_cars)+'car_p1_10iterations.txt',rho_vel_phi_stop30car_p1, fmt='%3.8f')
    
#%%
if __name__=='__main__':
	import multiprocessing as mp
	pool=mp.Pool(processes=4)
	results=[pool.apply_async(run_sim,args=(num_stopped_cars ,process_num))\
	for num_stopped_cars,process_num in zip([0,10,20,30],range(4))]
	#the_results=np.concatenate([p.get() for p in results])
	pool.close()