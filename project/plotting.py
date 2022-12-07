import numpy as np 
import matplotlib.pyplot as plt

leap_p = np.loadtxt("leapfrog_periodic.csv")
leap_np = np.loadtxt("leapfrog_NONperiodic.csv")

rk4_p = np.loadtxt("rk4_periodic.csv")
rk4_np = np.loadtxt("rk4_NONperiodic.csv")

leap_steps = np.arange(3,500)
rk4_steps = (leap_steps)

plt.title("Periodic BCs")
plt.plot(leap_steps, leap_p[:len(leap_steps)], label = "Leapfrog")
plt.plot(rk4_steps, rk4_p[:len(rk4_steps)], label = "RK4")
plt.xlabel("Number of steps")
plt.ylabel("Number total energy")
plt.legend()
plt.show 
plt.savefig("periodic_enery_plots.png")
plt.clf()

plt.title("Non-periodic BCs")
plt.plot(leap_steps[:len(rk4_np)], leap_np[:len(rk4_np)], label = "Leapfrog")
plt.plot(rk4_steps[:len(rk4_np)], rk4_np, label = "RK4")
plt.xlabel("Number of steps")
plt.ylabel("Number total energy")
plt.legend()
plt.show 
plt.savefig("non_periodic_enery_plots.png")