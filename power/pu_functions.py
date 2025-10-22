import numpy as np
import cmath

deg = np.pi/180

def print_base(base_name, base_val, units):
    print(f"{base_name} = {base_val} {units}")

def print_zpu(z_name, r_val, x_val, units):
    print(f"{z_name} = {round(r_val,3)} + j{round(x_val,3)} {units}")

def calc_vpu(v_name, v_pu, i_pu, z_pu):
    vpu = v_pu + i_pu * z_pu
    print(f"{v_name} = {round(np.abs(vpu),3)} < {round(np.angle(vpu,deg=True),3)}° pu")
    return vpu

def calc_ipu(i_name,i_pu):
    print(f"{i_name} = {round(np.abs(i_pu),3)} < {round(np.angle(i_pu,deg=True),3)}° pu")

def calc_actualv(v_name,v_level,Vpu):
    v_actual = v_level * Vpu
    print(f"{v_name} = {round(np.abs(v_actual/1000),3)} < {round(np.angle(v_actual, deg=True),3)}° kV")
    return v_actual

def calc_actuali(I_name,I_base,Ipu):
    I_actual = I_base * Ipu
    print(f"{I_name} = {round(np.abs(I_actual/1000),3)} < {round(np.angle(I_actual, deg=True),3)}° kA")
    return I_actual

def pf_correction(pf_init, correction):
    pfangle_init = np.arccos(pf_init)

    if correction == 0:
        return pf_init

    pfangle_target = pfangle_init * (1-correction)
    pf_corrected = np.cos(pfangle_target)

    return pf_corrected

def XFMR_phase(Vpu, connection):
    if connection == "Y-Δ":
        angle = 30
    elif connection == "Δ-Y":
        angle = -30
    return Vpu * cmath.exp(1j * angle * deg)

def change_base_vals(S_base_new, V_base1_new, V_base2_new, V_base4_new, V_base6_new):
    S_base = S_base_new
    V_base1 = V_base1_new
    V_base2 = V_base2_new
    V_base4 = V_base4_new
    V_base6 = V_base6_new
    return S_base, V_base1, V_base2, V_base4, V_base6

