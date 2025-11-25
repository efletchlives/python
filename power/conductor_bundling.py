import numpy as np
import cb_functions

# cardinal ACSR specs
Rc = 0.1128 # Ω/mi
GMR = 0.0403 # ft
GMD = cb_functions.GMD()

N = [2,3,4]
GMR_bnd, Ra, Xa, Za, Ya, R, X, Z, Y, z_pu, y_pu = [],[],[],[],[],[],[],[],[], [], []


# 2. determine Z & Y for different configs
# N = 2, 3, 4
for i in range(len(N)):
    # calculate GMR of bundle of N conductors
    gmr_bundle = cb_functions.GMR_bundle(GMR, N[i])
    GMR_bnd.append(gmr_bundle)

    # calculate series resistance per mi, series reactance per mi, series impedance per mi
    ra, xa, za = cb_functions.Za(60, Rc, GMD, GMR_bnd[i],N[i])
    Ra.append(ra)
    Xa.append(xa)
    Za.append(za)

    # calculate shunt admittance
    ya = cb_functions.Ya(60, GMD, GMR_bnd[i])
    Ya.append(ya)

    # calculate actual resistance, reactance, admittance
    r,x,z,y = cb_functions.actual(Ra[i],Xa[i],Za[i],Ya[i])
    R.append(r)
    X.append(x)
    Z.append(z)
    Y.append(y)

    z_pu_val, y_pu_val = cb_functions.pu_values(z, y)
    z_pu.append(z_pu_val)
    y_pu.append(y_pu_val)

    # print statements
    print(f'\nFor N = {N[i]}:')
    print(f'series resistance per mile: {Ra[i]:.2f} Ω/mi')
    print(f'series reactance per mile: j{Xa[i]:.2f} Ω/mi')
    print(f'shunt admittance per mile: j{Ya[i]} mS/mi')

    print(f'\nactual series resistance: {R[i]:.2f} Ω')
    print(f'actual series reactance: {X[i]:.2f} Ω')
    print(f'actual shunt admittance: {Y[i]} mS')

    print(f'\nper unit impedance: {z_pu[i]:.5f} pu')
    print(f'per unit admittance: {y_pu[i]:5f} pu')


# 3. calculate per unit voltages, physical currents and power loss
import math
S_base = 100 # MVA
S_load = 100 # MVA
pf = 0.9

phi_deg = math.degrees(math.acos(pf))
phi = math.radians(phi_deg)

S_load_pu = (S_load/S_base) * np.exp(1j*phi)
V_s = 1.0 + 0j

# subconductors: N = 2 
X_eq = 0.1 + X[0] + 0.1
Z[0] = R[0] + 1j*X_eq

    