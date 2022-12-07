import numpy as np
import numba as nb
from matplotlib import pyplot as plt
import numpy.fft
import imageio 


#This function assign to each assigns to every position the Left Top Nearest Grid Point (LT_NGP) by
#using the floor function on each position
def get_LT_NGP(ngrid, gridWidth, pos, periodic):

    if periodic == True:
        LT_NGP = np.floor(pos/gridWidth)%ngrid
    else:
        LT_NGP = np.floor(pos/gridWidth)
    return LT_NGP


#This function compute the density of each square of the grid by adding all the masses
#within it and dividing it by the area of a grid square.

#The Left Top cordinate of each particle is stored in a (npart x 2)
#We loop trough this array an increment the corresponding position 
#of in the rho matrix by with the mass of the particle 
@nb.njit
def get_rho(ngrid, gridWidth, LT_NGP, m, periodic):

    mass = m.copy()

    if periodic == True:

        rho = np.zeros((ngrid,ngrid))
        for i in range(LT_NGP.shape[0]):
            gridCord  = LT_NGP[i]
            rho[int(gridCord[0])][int(gridCord[1])] += mass[i]

#The non periodic case has a padding all around the mesh of a width of ngrid/2
#The big if statement with ngrid/2, 3*ngrid/2... at line 41 gets ride of every particles in the padding zone.
#This padding avoids the wrap around effects from the convolution 
    else:

        rho = np.zeros((ngrid*2,ngrid*2))
        for i in range(LT_NGP.shape[0]):
            gridCord  = LT_NGP[i]
            if not(gridCord[0] < ngrid/2 or gridCord[0] > 3*ngrid/2 or gridCord[1] < ngrid/2 or gridCord[1]> 3*ngrid/2): 
                rho[int(gridCord[0])][int(gridCord[1])] += mass[i]

    
    rho /= gridWidth**2
    
    return rho


#Moddified get_rho to accomodate for one particle only
#It was easier to do that then make if statements
def get_rho_1body(ngrid, gridWidth, LT_NGP, m, periodic):

    if periodic == True:

        rho = np.zeros((ngrid,ngrid))
        rho[int(LT_NGP[0])][int(LT_NGP[1])] += m
    else:
        rho = np.zeros((ngrid*2,ngrid*2))
        if not(gridCord[0] < ngrid/2 or gridCord[0] > 3*ngrid/2 or gridCord[1] < ngrid/2 or gridCord[1]> 3*ngrid/2):
            rho[int(gridCord[0])][int(gridCord[1])] += m
        else:
            print("Particle went out of bounds")
    
    rho /= gridWidth**2
    
    return rho


#I ripped this off from Prof. Sievers. I couldnt find a better kernel. WE
#need a kernel because the potential is the convolution of the density with the kernel
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


#This function simply takes the convolution between the kernel and the density to
#get the potential 
def get_potential(rho,kernel,ngrid,periodic):

    rhoFT= np.fft.rfft2(rho)
    kernelFT = np.fft.rfft2(kernel)
    if periodic:
        pot = np.fft.irfft2(rhoFT*kernelFT,[ngrid,ngrid])
    else:
        pot = np.fft.irfft2(rhoFT*kernelFT,[ngrid*2,ngrid*2])
    return pot 


#We use the same technique as we did to get the density, but this time to get the force
#The force is computed using a simple central difference
@nb.njit 
def get_force(ngrid,gridWidth, LT_NGP,pot,periodic):

    f = np.zeros((LT_NGP.shape[0],2))

    for i in range(LT_NGP.shape[0]):
        x  = int(LT_NGP[i][0])
        y =  int(LT_NGP[i][1])

        if periodic:
            f[i][0] = (pot[(x+1) % ngrid][y]-pot[(x-1) % ngrid][y])/(2*gridWidth)
            f[i][1] = (pot[x][(y+1) % ngrid]-pot[x][(y-1) % ngrid])/(2*gridWidth)

#Again for the the non periodic case we ignore everything that is in the padding zone
        else:
            if (x < ngrid/2 or x > 3*ngrid/2 or y < ngrid/2 or y > 3*ngrid/2 ):
                f[i][0] = 0
                f[i][1] = 0
            else:
                f[i][0] = (pot[(x+1)][y]-pot[(x-1)][y])/(2*gridWidth)
                f[i][1] = (pot[x][(y+1)]-pot[x][(y-1)])/(2*gridWidth)
    return f

#Get force adapted for single body problem
@nb.njit 
def get_force_1body(ngrid,gridWidth, LT_NGP,pot,periodic):

    f = np.zeros(2)

    x  = int(LT_NGP[0])
    y =  int(LT_NGP[1])

    if periodic:
        f[0] = (pot[(x+1) % ngrid][y]-pot[(x-1) % ngrid][y])/(2*gridWidth)
        f[1] = (pot[x][(y+1) % ngrid]-pot[x][(y-1) % ngrid])/(2*gridWidth)
    else:
        if (x < ngrid/2 or x > 3*ngrid/2 or y < ngrid/2 or y > 3*ngrid/2 ):
            f[0] = 0
            f[1] = 0
        else:
            f[0] = (pot[(x+1)][y]-pot[(x-1)][y])/(2*gridWidth)
            f[1] = (pot[x][(y+1)]-pot[x][(y-1)])/(2*gridWidth)
    return f

#This function takes a basic leapfrog step 
def take_frog_step(pos,v,f,dt):
    pos[:]=pos[:]+dt*v
    v[:]=v[:]+f*dt
    return pos, v

#The next two functions are for the rk4 and are also basically stolen from Prof. Sievers
#The implementation was just more clever than anything I came up with. 
#It stack the the postion and velocity in one array
#That get feed to get_derivs wich outputs the velocity and force in one array
#This get feed to get_derivs again 
#That happens a few time accordingly with the rk4 scheme, which generate all the ks needed
#to update the positions and velocities
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


#Just a class that contains all the information on the state of the system 
class particles:
    def __init__(self,npart=10000,ngrid=500, dx = 1, dt =0.05, soft=2, periodic=True):
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


#Gaussian initial conditions
    def ics_gauss(self):
        self.pos[:]=np.random.randn(self.npart,2)*(self.ngrid/20)+self.ngrid/2
        self.m[:]=1
        self.v[:]=0

#Initial conditions for Q1: Static single body 
    def ics_1body(self):
        self.npart = 1
        self.ngrid = 50
        if self.periodic:
            self.pos = np.array([25.,25.])
        else:
            self.pos = np.array([45.,45.])
        self.v = np.array([0.,0.])
        self.m = 1

#Intial conditions for Q2: Two orbiting bodies 
    def ics_2body(self):
        self.npart = 2
        self.ngrid = 50
        if self.periodic:
            self.pos = np.array([[20.,25.],[30.,25.]])
        else:
            self.pos = np.array([[40.,45.],[50.,45.]])
        self.v = np.array([[0.,-0.5],[0.0,0.5]])
        self.m[:] = 8

#Uniform distribution of particles all accross the mes
#Note that this one use a seed for the rng, which kept the same
#same for all the gif in uploaded to the repo 
    def ics_uniform(self):
        rng = np.random.default_rng(123456789123456789)
        if self.periodic:
            self.pos[:]=rng.random((self.npart,2))*self.ngrid
        else:
            self.pos[:]=rng.random((self.npart,2))*self.ngrid+self.ngrid/2
        self.m[:]=1
        self.v[:]=0