import numpy as np
import numba as nb
from matplotlib import pyplot as plt
import numpy.fft
import imageio 
from routines_and_ics import*


name = "1body"

parts=particles(dt =0.02)
parts.ics_1body()
parts.kernel = get_kernel(parts.ngrid, parts.soft, parts.periodic)

plt.ion()
osamp=3
fig = plt.figure()
ax = fig.add_subplot(111)
NumberOfIterations = 50

frames = []
energy = np.zeros(NumberOfIterations)

for i in range(NumberOfIterations):
    for j in range(osamp):
        parts.LT_NGP = get_LT_NGP(parts.ngrid, parts.gridWidth, parts.pos, parts.periodic)
        parts.rho = get_rho_1body(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.m,parts.periodic)
        parts.pot = get_potential(parts.rho, parts.kernel, parts.ngrid , parts.periodic)
        parts.f = get_force_1body(parts.ngrid, parts.gridWidth, parts.LT_NGP, parts.pot,parts.periodic)
        parts.pos, parts.v = take_frog_step(parts.pos, parts.v, parts.f, parts.dt)
        #parts.pos, parts.v = take_rk4_step(parts.ngrid,parts.gridWidth, parts.LT_NGP, parts.pos,parts.v, parts.pot,parts.periodic, parts.dt)
    kin=np.sum(parts.v**2)
    pot=np.sum(parts.rho*parts.pot)
    E = kin+pot
    print(E)
    plt.clf()
    plt.imshow(parts.rho);#,vmin=25,vmax=75)
    plt.title("frame %i" %i)
    plt.colorbar()
    plt.pause(0.0001)

    energy[i] += E

    plt.savefig("frames/"+str(i)+".png")
    plt.close()
    frames.append("frames/"+str(i)+".png")
    os.remove

np.savetxt(name +".csv", energy)

with imageio.get_writer(name +".gif", mode="I") as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

      

