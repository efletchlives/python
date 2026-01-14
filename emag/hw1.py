import numpy as np 
import matplotlib.pyplot as plt


r = 2600 # ohms

# f(x) = a*cos(bx + c) + d
# a. find period
T = (2*np.pi)/(0.03*np.pi) # seconds
print(f'period is {T:.2f} sec')

t = np.linspace(0,5*T,1000) # seconds
v = 1.3*np.cos(0.03*np.pi*t) # mV

# b. calculate current
i = (v*10e3)/r # uA

# c. calculate instantaneous power
p = i*v # nW

# d. energy consumed by resistor in 0.54 periods starting at t = 0
# E = p*t

E = 1.3*np.cos(0.03*np.pi*0.54*T) * (1.3/2600)*np.cos(0.03*np.pi*0.54*T) * 0.54*T
print(f'the energy consumed in 0.54 periods = {E}')


# plot voltage, current, and power over 5 periods
# voltage
plt.figure()
plt.plot(t,v,color='blue',linewidth=2)
plt.title("Time vs Voltage Plot")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (mV)")
plt.savefig("/workspaces/python/emag/voltage.png",dpi=300)

# current
plt.figure()
plt.plot(t,i,color='red',linewidth=2)
plt.title("Time vs Current Plot")
plt.xlabel("Time (seconds)")
plt.ylabel("Current (uA)")
plt.savefig("/workspaces/python/emag/current.png",dpi=300)

# power
plt.figure()
plt.plot(t,p,color='orange',linewidth=2)
plt.title("Time vs Power Plot")
plt.xlabel("Time (seconds)")
plt.ylabel("Power (nW)")
plt.savefig("/workspaces/python/emag/power.png",dpi=300)
