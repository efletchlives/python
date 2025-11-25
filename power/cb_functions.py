import numpy as np
# constants
ε0 = 8.85e-12
miles = 200
v_base = 345 # kV
s_base = 100 # MVA

Dab = 32.8 # ft
Dbc = 32.8 # ft
Dca = 46.4 # ft

def GMR_bundle(GMR, bundle_num):
    if bundle_num == 2:
        return np.sqrt(GMR * 1)
    
    elif bundle_num == 3:
        return (GMR * 1.5**2)**(1/3)
    
    elif bundle_num == 4:
        return (GMR * 1.5**3)**(1/4)

def GMD():
    return (Dab * Dbc * Dca)**(1/3)

def Za(freq, Rc, GMD, gmr_bundle, bundle_num):
    Ra = Rc/bundle_num # Ω/mi
    Xa = 2*np.pi *freq * 2e-7*np.log(GMD/gmr_bundle)*1609 # H/mi

    Za = Ra + 1j * Xa
    return Ra, Xa, Za

def Ya(freq, GMD, gmr_bundle):
    return 2*np.pi*freq * (2*np.pi*ε0)/(np.log(GMD/gmr_bundle))*1609 # S/mi

def actual(Ra, Xa, Za, Ya):
    R,X,Z = Ra*miles, Xa*miles, Za*miles
    Y = Ya*miles*np.power(10,3)
    return R,X,Z,Y

def pu_values(z_actual, y_actual):
    Z_base = (v_base**2) / (s_base)
    Y_base = 1/Z_base

    z_pu = z_actual/Z_base
    y_pu = y_actual * Z_base
    return z_pu, y_pu