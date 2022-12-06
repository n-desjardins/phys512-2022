import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy.fft
import imageio 
import os

def get_LT_NGP(ngrid, gridWidth, pos, periodic):

    if periodic == True:
        LT_NGP = np.floor(pos/gridWidth)%ngrid
    else:
        LT_NGP = np.floor(pos/gridWidth)
    return LT_NGP


@nb.njit
def get_rho(ngrid, gridWidth, LT_NGP, m, periodic):

    mass = m.copy()

    if periodic == True:

        rho = np.zeros((ngrid,ngrid))
        for i in range(LT_NGP.shape[0]):
            gridCord  = LT_NGP[i]
            rho[int(gridCord[0])][int(gridCord[1])] += mass[i]
    else:


        rho = np.zeros((ngrid*2,ngrid*2))
        for i in range(LT_NGP.shape[0]):
            gridCord  = LT_NGP[i]
            if not(gridCord[0] < ngrid/2 or gridCord[0] > 3*ngrid/2 or gridCord[1] < ngrid/2 or gridCord[1]> 3*ngrid/2):
                rho[int(gridCord[0])][int(gridCord[1])] += mass[i]

    
    rho /= gridWidth**2
    
    return rho



def get_kernel(ngrid,r0,periodic):
    if not(periodic):
        x=np.fft.fftfreq(ngrid*2)*ngrid*2
        rsqr=np.outer(np.ones(ngrid*2),x**2)
    else:
        x=np.fft.fftfreq(ngrid)*ngrid
        rsqr=np.outer(np.ones(ngrid),x**2)
    rsqr=rsqr+rsqr.T
    rsqr[rsqr<r0**2]=r0**2
    kernel=rsqr**-0.5
    return kernel



def get_potential(rho,kernel,ngrid,periodic):

    rhoFT= np.fft.rfft2(rho)
    kernelFT = np.fft.rfft2(kernel)
    if periodic:
        pot = np.fft.irfft2(rhoFT*kernelFT,[ngrid,ngrid])
    else:
        pot = np.fft.irfft2(rhoFT*kernelFT,[ngrid*2,ngrid*2])
    return pot 


@nb.njit 
def get_force(ngrid,gridWidth, LT_NGP,pot,periodic):

    f = np.zeros((LT_NGP.shape[0],2))

    for i in range(LT_NGP.shape[0]):
        x  = int(LT_NGP[i][0])
        y =  int(LT_NGP[i][1])

        if periodic:
            f[i][0] = (pot[(x+1) % ngrid][y]-pot[(x-1) % ngrid][y])/(2*gridWidth)
            f[i][1] = (pot[x][(y+1) % ngrid]-pot[x][(y-1) % ngrid])/(2*gridWidth)
        else:
            if (x < ngrid/2 or x > 3*ngrid/2 or y < ngrid/2 or y > 3*ngrid/2 ):
                f[i][0] = 0
                f[i][1] = 0
            else:
                f[i][0] = (pot[(x+1)][y]-pot[(x-1)][y])/(2*gridWidth)
                f[i][1] = (pot[x][(y+1)]-pot[x][(y-1)])/(2*gridWidth)
    return f


def take_frog_step(pos,v,f,dt):
    pos[:]=pos[:]+dt*v
    v[:]=v[:]+f*dt
    return pos, v

def get_derivs(ngrid,gridWidth, xx,pot,periodic):
    nn=xx.shape[0]//2
    x=xx[:nn,:]
    v=xx[nn:,:]
    f=get_force(ngrid,gridWidth, x, pot,periodic)
    return np.vstack([v,f])
    
def take_rk4_step(ngrid,gridWidth, LT_NGP,pos,v, pot,periodic, dt):

    xx=np.vstack([LT_NGP,v])

    k1=get_derivs(ngrid,gridWidth, xx,pot,periodic)
    k2=get_derivs(ngrid,gridWidth, xx+k1*dt/2,pot,periodic)
    k3=get_derivs(ngrid,gridWidth, xx+k2*dt/2,pot,periodic)
    k4=get_derivs(ngrid,gridWidth, xx+k3*dt,pot,periodic)
    
    tot=(k1+2*k2+2*k3+k4)/6

    if periodic:
        nn=pos.shape[0]
        pos=pos+(tot[:nn,:])*dt
        v=v+tot[nn:,:]*dt
    else:
        nn=pos.shape[0]
        pos=pos+tot[:nn,:]*dt
        v=v+tot[nn:,:]*dt
    return pos,v

class particles:
    def __init__(self,npart=10000,ngrid=1000, dx = 1, dt =0.05, soft=1,periodic=True):
        self.pos=np.empty([npart,2])
        self.m=np.empty(npart)
        self.f=np.empty([npart,2])
        self.v=np.empty([npart,2])
        self.kernel=[]
        self.soft=soft
        self.periodic=periodic
        self.npart=npart
        self.ngrid=ngrid
        self.gridWidth = dx
        self.dt = dt
        self.LT_NGP=np.empty([self.npart,2])
        if self.periodic:
            self.rho=np.empty([self.ngrid,self.ngrid])
            self.pot=np.empty([self.ngrid,self.ngrid])
        else:
            self.rho=np.empty([self.ngrid*2,self.ngrid*2])
            self.pot=np.empty([self.ngrid*2,self.ngrid*2])

    
    def ics_gauss(self):
        self.pos[:]=np.random.randn(self.npart,2)*(self.ngrid/20)+self.ngrid/2
        self.m[:]=1
        self.v[:]=0

    def ics_2gauss(self):
        #self.pos[:]=np.random.randn(self.npart,2)*(self.ngrid/12)+self.ngrid/2
        #self.pos[:self.npart//2,0]=self.pos[:self.npart//2,0]-self.ngrid/5
        #self.pos[self.npart//2:,0]=self.pos[self.npart//2:,0]+self.ngrid/5
        #self.pos[:] = self.pos[:]%self.ngrid
        #self.m[:] =1
        #self.v[:]=0

        self.pos[:]=np.random.randn(self.npart,2)*(self.ngrid/6)+self.ngrid
        self.pos[:self.npart//2,0]=self.pos[:self.npart//2,0]-self.ngrid
        self.pos[self.npart//2:,0]=self.pos[self.npart//2:,0]+self.ngrid
        self.pos[:] = self.pos[:]%self.ngrid
        self.m[:] =1
        self.v[:]=0
    
    def ics_2body(self):
        self.pos = np.array([[20.,25.],[30.,25.]])
        self.v = np.array([[0.,-0.5],[0.0,0.5]])
        self.m[:] = 8

        #self.pos = np.array([[45.,45.],[55.,45.]])
        #self.v = np.array([[0.,0.5],[0.0,-0.5]])
        #self.m[:] = 1

    def ics_uniform_periodic(self):
        rng = np.random.default_rng(123456789123456789)
        self.pos[:]=rng.random((self.npart,2))*self.ngrid
        self.m[:]=1
        self.v[:]=0

    def ics_uniform_non_periodic(self):
        rng = np.random.default_rng(123456789123456789)
        self.pos[:]=rng.random((self.npart,2))*self.ngrid+self.ngrid/2
        self.m[:]=1
        self.v[:]=0



    

#parts=particles(npart=2,ngrid=50, dx = 1, dt =0.02, soft=2,periodic=True)
parts=particles(npart=100000,ngrid=500, dx = 1, dt =0.01, soft=2,periodic=True)


#parts.ics_gauss()
#parts.ics_uniform_non_periodic()
parts.ics_uniform_periodic()
#parts.ics_2body()
parts.kernel = get_kernel(parts.ngrid, parts.soft, parts.periodic)

plt.ion()
osamp=3

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
NumberOfIterations = 30

name = "rk4_periodic"

#frames = []
energy = np.zeros(NumberOfIterations)

for i in range(NumberOfIterations):
    for j in range(osamp):
        parts.LT_NGP = get_LT_NGP(parts.ngrid, parts.gridWidth, parts.pos, parts.periodic)
        parts.rho = get_rho(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.m,parts.periodic)
        parts.pot = get_potential(parts.rho, parts.kernel, parts.ngrid , parts.periodic)
        parts.f = get_force(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.pot,parts.periodic)
        #parts.pos, parts.v = take_frog_step(parts.pos, parts.v, parts.f, parts.dt)
        parts.pos, parts.v = take_rk4_step(parts.ngrid,parts.gridWidth, parts.LT_NGP, parts.pos,parts.v, parts.pot,parts.periodic, parts.dt)
    kin=np.sum(parts.v**2)
    pot=np.sum(parts.rho*parts.pot)
    E = kin+pot
    print(E)
    plt.clf()
    plt.imshow(parts.rho);#,vmin=25,vmax=75)
    plt.colorbar()
    plt.pause(0.0001)

    energy[i] += E

    #plt.savefig("frames/"+str(i)+".png")
    #plt.close()
    #frames.append("frames/"+str(i)+".png")

np.savetxt(name +".csv", energy)

#with imageio.get_writer("leapfrog_periodic.gif", mode="I") as writer:
    #for frame in frames:
        #image = imageio.imread(frame)
       #writer.append_data(image)
      

