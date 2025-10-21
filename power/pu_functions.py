import numpy as np

def print_base(base_name, base_val, units):
    print(f"{base_name} = {base_val} {units}")

def print_zpu(z_name, r_val, x_val, units):
    print(f"{z_name} = {round(r_val,3)} + j{round(x_val,3)} {units}")

def calc_vpu(v_name, v_pu, i_pu, z_pu):
    vpu = v_pu + i_pu * z_pu
    print(f"{v_name} = {round(np.abs(vpu),3)} < {round(np.angle(vpu),3)} pu")
    return vpu

def calc_ipu(i_name,i_pu):
    print(f"{i_name} = {round(np.abs(i_pu),3)} < {round(np.angle(i_pu),3)} pu")

def calc_actualv(v_name,v_level,Vpu):
    v_actual = v_level * Vpu
    print(f"{v_name} = {round(np.abs(v_actual/1000),3)} < {round(np.angle(v_actual),3)} kV")
    return v_actual

def pf_correction(pf_init, correction):
    pfangle_init = np.arccos(pf_init)

    if correction == 0:
        return pf_init

    pfangle_target = pfangle_init * (1-correction)
    pf_corrected = np.cos(pfangle_target)

    return pf_corrected
