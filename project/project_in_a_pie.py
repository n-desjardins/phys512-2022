import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import numpy.fft

def get_LT_NGP(ngrid, gridWidth, pos):
    LT_NGP = np.floor(pos/gridWidth)%ngrid
    return LT_NGP




"""""
    if (type(m) != int):
        for i in range(LT_NGP.shape[0]):
            gridCord = LT_NGP[i]
            rho[int(gridCord[0])][int(gridCord[1])] += m[i]
    else:
        for i in range(LT_NGP.shape[0]):
            gridCord  = LT_NGP[i]
            rho[int(gridCord[0])][int(gridCord[1])] += m
    """


@nb.njit
def get_rho(ngrid, gridWidth, LT_NGP, m):

    rho = np.zeros((ngrid,ngrid))

    for i in range(LT_NGP.shape[0]):
        gridCord  = LT_NGP[i]
        rho[int(gridCord[0])][int(gridCord[1])] += m
    
    rho /= gridWidth**2
    
    return rho



def get_kernel(ngrid,r0):
    x=np.fft.fftfreq(ngrid)*ngrid
    rsqr=np.outer(np.ones(ngrid),x**2)
    rsqr=rsqr+rsqr.T
    rsqr[rsqr<r0**2]=r0**2
    kernel=rsqr**-0.5
    return kernel



def get_potential(rho,kernel,ngrid):

    rhoFT= np.fft.rfft2(rho)
    kernelFT = np.fft.rfft2(kernel)
    pot = np.fft.irfft2(rhoFT*kernelFT,[ngrid,ngrid])
    return pot 


@nb.njit 
def get_force(ngrid,gridWidth, LT_NGP,pot):

    f = np.zeros((LT_NGP.shape[0],2))

    for i in range(LT_NGP.shape[0]):
        x  = int(LT_NGP[i][0])
        y =  int(LT_NGP[i][1])

        f[i][0] = (pot[(x+1) % ngrid][y]-pot[(x-1) % ngrid][y])/(2*gridWidth)
        f[i][1] = (pot[x][(y+1) % ngrid]-pot[x][(y-1) % ngrid])/(2*gridWidth)
    return f


def take_step(pos,v,f,dt):
    pos[:]=pos[:]+dt*v
    v[:]=v[:]+f*dt
    return pos, v


class particles:
    def __init__(self,npart=10000,ngrid=1000, dx = 1, dt =0.02, soft=1,periodic=True):
        self.pos=np.empty([npart,2])
        self.m=np.empty(npart)
        self.f=np.empty([npart,2])
        self.v=np.empty([npart,2])
        self.kernel=[]
        self.npart=npart
        self.ngrid=ngrid
        self.gridWidth = dx
        self.dt = dt
        self.LT_NGP=np.empty([self.npart,2])
        self.rho=np.empty([self.ngrid,self.ngrid])
        self.pot=np.empty([self.ngrid,self.ngrid])
        
        self.soft=soft
        self.periodic=periodic

    def ics_2gauss(self):
        self.pos[:]=np.random.randn(self.npart,2)*(self.ngrid/12)+self.ngrid/2
        self.pos[:self.npart//2,0]=self.pos[:self.npart//2,0]-self.ngrid/5
        self.pos[self.npart//2:,0]=self.pos[self.npart//2:,0]+self.ngrid/5
        self.pos[:] = self.pos[:]%self.ngrid
        self.m =1
        self.v[:]=0
    
    def ics_2body(self):
        self.pos = np.array([[20.,25.],[30.,25.]])
        self.v = np.array([[0.,-0.5],[0.0,0.5]])
        self.m = 8



    

parts=particles(npart=2,ngrid=50, dx = 1, dt =0.02, soft=2,periodic=True)


#parts.ics_2gauss()
parts.ics_2body()
parts.kernel = get_kernel(parts.ngrid, parts.soft)

plt.ion()
osamp=3

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
crap=ax.imshow(parts.rho[:parts.ngrid,:parts.ngrid]**0.5)

for i in range(1500):
    for j in range(osamp):
        parts.LT_NGP = get_LT_NGP(parts.ngrid, parts.gridWidth, parts.pos)
        parts.rho = get_rho(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.m)
        parts.pot = get_potential(parts.rho, parts.kernel, parts.ngrid )
        parts.f = get_force(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.pot)
        parts.pos, parts.v = take_step(parts.pos, parts.v, parts.f, parts.dt)
    kin=np.sum(parts.v**2)
    pot=np.sum(parts.rho*parts.pot)
    print(kin,pot,kin-0.5*pot)
    plt.clf()
    plt.imshow(parts.rho*100)#,vmin=0.9,vmax=1.1)
    plt.colorbar()


    crap.set_data(parts.rho[:parts.ngrid,:parts.ngrid]**0.5)
    plt.pause(0.0001)