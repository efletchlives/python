import numpy as np

import pu_functions

# ---------------------------- 1 & 2. --------------------------------
print("1 & 2")
# set the base values (given)
S_base = 100e6 # 100MVA
pu_functions.print_base("S_base",round(S_base/(1e6)),"MVA")
V_base1 = 20e3 # 20kV
pu_functions.print_base("V_base1",round(V_base1/(1e3)),"kV")
V_base2 = 230e3 # 230kV
pu_functions.print_base("V_base2",round(V_base2/(1e3)),"kV")
V_base4 = 12.47e3 # 12.47kV
pu_functions.print_base("V_base4",round(V_base4/(1e3),2),"kV")
V_base6 = 4.16e3 # 4.16kV
pu_functions.print_base("V_base6",round(V_base6/(1e3),2),"kV")

Z_base2 = V_base2**2/(S_base)
pu_functions.print_base("Z_base2",round(Z_base2),"ohms")
Z_base4 = V_base4**2/(S_base)
pu_functions.print_base("Z_base4",round(Z_base4,3),"ohms")
Z_base6 = V_base6**2/(S_base)
pu_functions.print_base("Z_base6",round(Z_base6,3),"ohms")

# transformer per unit values (given: 0.1pu)
x_T1pu = 0.1*(S_base/100e6)
pu_functions.print_base("Xpu_T1",x_T1pu,"pu")
x_T2pu = 0.1*(S_base/75e6)
pu_functions.print_base("Xpu_T2",round(x_T2pu,2),"pu")
x_T3pu = 0.1*(S_base/25e6)
pu_functions.print_base("Xpu_T3",x_T3pu,"pu")

# line per unit values (given)
R_Ln1, X_Ln1 = 0.0376, 0.5277
R_Ln2, X_Ln2 = 0.4576, 1.0780

len1 = 100 # miles
len2 = 2000/5280 # ft to miles
R_Ln1pu = (R_Ln1*len1)/Z_base2
X_Ln1pu = (X_Ln1*len1)/Z_base2
Z_Ln1pu = (R_Ln1 + 1j*X_Ln1)*len1/Z_base2
pu_functions.print_zpu("Zpu_Line1",round(R_Ln1pu,3),round(X_Ln1pu,3),"pu")

R_Ln2pu = (R_Ln2*len2)/Z_base4
X_Ln2pu = (X_Ln2*len2)/Z_base4
Z_Ln2pu = (R_Ln2 + 1j*X_Ln2)*len2/Z_base4
pu_functions.print_zpu("Zpu_Line2",round(R_Ln2pu,3),round(X_Ln2pu,3),"pu")

# load per unit values Z_Load = V^2/S < (cos^-1(pf))

# Load 1 
# given values
S_Ld1 = 50e6 # 50MVA
pf_Ld1 = 0.85

# find load 1 per unit impedance
phase_Ld1 = np.arccos(pf_Ld1)
re_Ld1 = np.power(V_base4,2)/(S_Ld1) * np.cos(phase_Ld1)
im_Ld1 = np.power(V_base4,2)/(S_Ld1) * np.sin(phase_Ld1)
Z_Ld1 = re_Ld1 + 1j*im_Ld1
Z_Ld1pu = Z_Ld1/Z_base4
pu_functions.print_zpu("Zpu_Load1",round(re_Ld1,3),round(im_Ld1,3),"pu")

# Load 2
# given values
S_Ld2 = 20e6 # 20MVA
pf_Ld2 = 0.9

# find load 2 per unit impedance
phase_Ld2 = np.arccos(pf_Ld2)
re_Ld2 = np.power(V_base6,2)/S_Ld2 * np.cos(phase_Ld2)
im_Ld2 = np.power(V_base6,2)/S_Ld2 * np.sin(phase_Ld2)
Z_Ld2 = re_Ld2 + 1j*im_Ld2
Z_Ld2pu = Z_Ld2/Z_base6
pu_functions.print_zpu("Zpu_Load2",round(re_Ld2,3),round(im_Ld2,3),"pu\n")


# ---------------------------- 3 & 4 -----------------------------------
# assume 4kV operating voltage at Load 2
print("3.")
# Bus 6
V6_actual = 4e3
V6_pu = V6_actual/V_base6
print("per unit voltages:")
print(f"V6_pu = {round(V6_pu,3)} < 0 pu")

I6_pu = V6_pu/Z_Ld2pu

# other buses
V5_pu = pu_functions.calc_vpu("V5_pu", V6_pu, I6_pu, x_T3pu)
V4_pu = pu_functions.calc_vpu("V4_pu", V5_pu, I6_pu, Z_Ln2pu)

I_Ld1 = V4_pu/Z_Ld1pu
I3_pu = I6_pu + I_Ld1
V3_pu = pu_functions.calc_vpu("V3_pu", V4_pu, I3_pu, x_T2pu)
V2_pu = pu_functions.calc_vpu("V2_pu", V3_pu, I3_pu, Z_Ln1pu)
V1_pu = pu_functions.calc_vpu("V1_pu", V2_pu, I3_pu, x_T1pu)

print("per unit currents:")
pu_functions.calc_ipu("I6_pu",I6_pu)
pu_functions.calc_ipu("I5_pu",I6_pu)
pu_functions.calc_ipu("I4_pu",I6_pu)
pu_functions.calc_ipu("I3_pu",I3_pu)
pu_functions.calc_ipu("I2_pu",I3_pu)
pu_functions.calc_ipu("I1_pu",I3_pu)

# calculate actual voltages
print ('\n4.')
V6_actual = pu_functions.calc_actualv("V6",4.16e3,V6_pu)
V5_actual = pu_functions.calc_actualv("V5",12.47e3,V5_pu)
V4_actual = pu_functions.calc_actualv("V4",230e3,V4_pu)
V3_actual = pu_functions.calc_actualv("V3",230e3,V3_pu)
V2_actual = pu_functions.calc_actualv("V2",230e3,V2_pu)
V1_actual = pu_functions.calc_actualv("V1",20e3,V1_pu)

# calculate actual currents
# I6_actual 



# ------------------------------- 5 ------------------------------------
# employ a pf correction solution

# ----------- correction = 0 -------------
print('\ncase 1: correction = 0')
correction = 0
pf_Ld1_corr = pu_functions.pf_correction(pf_Ld1,correction)
pf_Ld2_corr = pu_functions.pf_correction(pf_Ld2,correction)

print(f'Load 1 PF: {pf_Ld1_corr}')
print(f'Load 2 PF: {pf_Ld2_corr}')

phase1 = np.arccos(pf_Ld1_corr)
phase2 = np.arccos(pf_Ld2_corr)

re_Ld1 = V_base4**2/S_Ld1 * np.cos(phase1)
im_Ld1 = V_base4**2/S_Ld1 * np.sin(phase1)
re_Ld2 = V_base6**2/S_Ld2 * np.cos(phase2)
im_Ld2 = V_base6**2/S_Ld2 * np.sin(phase2)
Z_Ld1 = (V_base4**2 / S_Ld1) * (np.cos(phase1) + 1j * np.sin(phase1))
Z_Ld2 = (V_base6**2 / S_Ld2) * (np.cos(phase2) + 1j * np.sin(phase2))
Z_Ld1pu = Z_Ld1 / Z_base4
Z_Ld2pu = Z_Ld2 / Z_base6

pu_functions.print_zpu("Zpu_Load1", re_Ld1, im_Ld1, "pu")
pu_functions.print_zpu("Zpu_Load2", re_Ld2, im_Ld2, "pu")

print('\nper unit voltages:')
V6_actual = 4e3
V6_pu = V6_actual / V_base6
print(f"V6_pu = {round(V6_pu,3)} < 0 pu")

I6_pu = V6_pu / Z_Ld2pu
V5_pu = pu_functions.calc_vpu("V5_pu", V6_pu, I6_pu, x_T3pu)
V4_pu = pu_functions.calc_vpu("V4_pu", V5_pu, I6_pu, Z_Ln2pu)
I_Ld1 = V4_pu / Z_Ld1pu
I3_pu = I6_pu + I_Ld1
V3_pu = pu_functions.calc_vpu("V3_pu", V4_pu, I3_pu, x_T2pu)
V2_pu = pu_functions.calc_vpu("V2_pu", V3_pu, I3_pu, Z_Ln1pu)
V1_pu = pu_functions.calc_vpu("V1_pu", V2_pu, I3_pu, x_T1pu)

print("\nper unit currents:")
pu_functions.calc_ipu("I6_pu", I6_pu)
pu_functions.calc_ipu("I5_pu", I6_pu)
pu_functions.calc_ipu("I4_pu", I6_pu)
pu_functions.calc_ipu("I3_pu", I3_pu)
pu_functions.calc_ipu("I2_pu", I3_pu)
pu_functions.calc_ipu("I1_pu", I3_pu)

print('actual voltages:')
V6_actual = pu_functions.calc_actualv("V6", 4.16e3, V6_pu)
V5_actual = pu_functions.calc_actualv("V5", 12.47e3, V5_pu)
V4_actual = pu_functions.calc_actualv("V4", 230e3, V4_pu)
V3_actual = pu_functions.calc_actualv("V3", 230e3, V3_pu)
V2_actual = pu_functions.calc_actualv("V2", 230e3, V2_pu)
V1_actual = pu_functions.calc_actualv("V1", 20e3, V1_pu)


# -------- correction = 0.5 -----------
print('\ncase 1: correction = 0.5')
correction = 0.5
pf_Ld1_corr = pu_functions.pf_correction(pf_Ld1,correction)
pf_Ld2_corr = pu_functions.pf_correction(pf_Ld2,correction)

print(f'Load 1 PF: {pf_Ld1_corr}')
print(f'Load 2 PF: {pf_Ld2_corr}')

phase1 = np.arccos(pf_Ld1_corr)
phase2 = np.arccos(pf_Ld2_corr)

re_Ld1 = V_base4**2/S_Ld1 * np.cos(phase1)
im_Ld1 = V_base4**2/S_Ld1 * np.sin(phase1)
re_Ld2 = V_base6**2/S_Ld2 * np.cos(phase2)
im_Ld2 = V_base6**2/S_Ld2 * np.sin(phase2)
Z_Ld1 = (V_base4**2 / S_Ld1) * (np.cos(phase1) + 1j * np.sin(phase1))
Z_Ld2 = (V_base6**2 / S_Ld2) * (np.cos(phase2) + 1j * np.sin(phase2))
Z_Ld1pu = Z_Ld1 / Z_base4
Z_Ld2pu = Z_Ld2 / Z_base6

pu_functions.print_zpu("Zpu_Load1", re_Ld1, im_Ld1, "pu")
pu_functions.print_zpu("Zpu_Load2", re_Ld2, im_Ld2, "pu")

print('\nper unit voltages:')
V6_actual = 4e3
V6_pu = V6_actual / V_base6
print(f"V6_pu = {round(V6_pu,3)} < 0 pu")

I6_pu = V6_pu / Z_Ld2pu
V5_pu = pu_functions.calc_vpu("V5_pu", V6_pu, I6_pu, x_T3pu)
V4_pu = pu_functions.calc_vpu("V4_pu", V5_pu, I6_pu, Z_Ln2pu)
I_Ld1 = V4_pu / Z_Ld1pu
I3_pu = I6_pu + I_Ld1
V3_pu = pu_functions.calc_vpu("V3_pu", V4_pu, I3_pu, x_T2pu)
V2_pu = pu_functions.calc_vpu("V2_pu", V3_pu, I3_pu, Z_Ln1pu)
V1_pu = pu_functions.calc_vpu("V1_pu", V2_pu, I3_pu, x_T1pu)

print("\nper unit currents:")
pu_functions.calc_ipu("I6_pu", I6_pu)
pu_functions.calc_ipu("I5_pu", I6_pu)
pu_functions.calc_ipu("I4_pu", I6_pu)
pu_functions.calc_ipu("I3_pu", I3_pu)
pu_functions.calc_ipu("I2_pu", I3_pu)
pu_functions.calc_ipu("I1_pu", I3_pu)

print('actual voltages:')
V6_actual = pu_functions.calc_actualv("V6", 4.16e3, V6_pu)
V5_actual = pu_functions.calc_actualv("V5", 12.47e3, V5_pu)
V4_actual = pu_functions.calc_actualv("V4", 230e3, V4_pu)
V3_actual = pu_functions.calc_actualv("V3", 230e3, V3_pu)
V2_actual = pu_functions.calc_actualv("V2", 230e3, V2_pu)
V1_actual = pu_functions.calc_actualv("V1", 20e3, V1_pu)


# -------- correction = 1 ------------
print('\ncase 1: correction = 1')
correction = 1
pf_Ld1_corr = pu_functions.pf_correction(pf_Ld1,correction)
pf_Ld2_corr = pu_functions.pf_correction(pf_Ld2,correction)

print(f'Load 1 PF: {pf_Ld1_corr}')
print(f'Load 2 PF: {pf_Ld2_corr}')

phase1 = np.arccos(pf_Ld1_corr)
phase2 = np.arccos(pf_Ld2_corr)

re_Ld1 = V_base4**2/S_Ld1 * np.cos(phase1)
im_Ld1 = V_base4**2/S_Ld1 * np.sin(phase1)
re_Ld2 = V_base6**2/S_Ld2 * np.cos(phase2)
im_Ld2 = V_base6**2/S_Ld2 * np.sin(phase2)
Z_Ld1 = (V_base4**2 / S_Ld1) * (np.cos(phase1) + 1j * np.sin(phase1))
Z_Ld2 = (V_base6**2 / S_Ld2) * (np.cos(phase2) + 1j * np.sin(phase2))
Z_Ld1pu = Z_Ld1 / Z_base4
Z_Ld2pu = Z_Ld2 / Z_base6

pu_functions.print_zpu("Zpu_Load1", re_Ld1, im_Ld1, "pu")
pu_functions.print_zpu("Zpu_Load2", re_Ld2, im_Ld2, "pu")

print('\nper unit voltages:')
V6_actual = 4e3
V6_pu = V6_actual / V_base6
print(f"V6_pu = {round(V6_pu,3)} < 0 pu")

I6_pu = V6_pu / Z_Ld2pu
V5_pu = pu_functions.calc_vpu("V5_pu", V6_pu, I6_pu, x_T3pu)
V4_pu = pu_functions.calc_vpu("V4_pu", V5_pu, I6_pu, Z_Ln2pu)
I_Ld1 = V4_pu / Z_Ld1pu
I3_pu = I6_pu + I_Ld1
V3_pu = pu_functions.calc_vpu("V3_pu", V4_pu, I3_pu, x_T2pu)
V2_pu = pu_functions.calc_vpu("V2_pu", V3_pu, I3_pu, Z_Ln1pu)
V1_pu = pu_functions.calc_vpu("V1_pu", V2_pu, I3_pu, x_T1pu)

print("\nper unit currents:")
pu_functions.calc_ipu("I6_pu", I6_pu)
pu_functions.calc_ipu("I5_pu", I6_pu)
pu_functions.calc_ipu("I4_pu", I6_pu)
pu_functions.calc_ipu("I3_pu", I3_pu)
pu_functions.calc_ipu("I2_pu", I3_pu)
pu_functions.calc_ipu("I1_pu", I3_pu)

print('actual voltages:')
V6_actual = pu_functions.calc_actualv("V6", 4.16e3, V6_pu)
V5_actual = pu_functions.calc_actualv("V5", 12.47e3, V5_pu)
V4_actual = pu_functions.calc_actualv("V4", 230e3, V4_pu)
V3_actual = pu_functions.calc_actualv("V3", 230e3, V3_pu)
V2_actual = pu_functions.calc_actualv("V2", 230e3, V2_pu)
V1_actual = pu_functions.calc_actualv("V1", 20e3, V1_pu)



# ------------------ 6 ---------------------------
# reflection on pf correction


# ------------------ 7 ---------------------------
# changing base values
