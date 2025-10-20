import numpy as np

# ------------------ 1 & 2. -------------------------
print("1 & 2")
# set the base values (given)
S_base = 100e6 # 100MVA
print(f"S_base = {round(S_base*1/np.power(10,6))} MVA")
V_base1 = 20e3 # 20kV
print(f"V_base1 = {round(V_base1*0.001)} kV")
V_base2 = 230e3 # 230kV
print(f"V_base2 = {round(V_base2*0.001)} kV")
V_base4 = 12.47e3 # 12.47kV
print(f"V_base4 = {V_base4*0.001} kV")
V_base6 = 4.16e3 # 4.16kV
print(f"V_base6 = {V_base6*0.001} kV")

Z_base2 = V_base2**2/(S_base)
print(f"Z_base2 = {round(Z_base2)} ohms")
Z_base4 = V_base4**2/(S_base)
print(f"Z_base4 = {round(Z_base4,3)} ohms")
Z_base6 = V_base6**2/(S_base)
print(f"Z_base6 = {round(Z_base6,3)} ohms")

# transformer per unit values (given: 0.1pu)
x_T1pu = 0.1*(S_base/100e6)
print(f"Xpu_T1 = {x_T1pu} pu")
x_T2pu = 0.1*(S_base/75e6)
print(f"Xpu_T2 = {round(x_T2pu,2)} pu")
x_T3pu = 0.1*(S_base/25e6)
print(f"Xpu_T3 = {x_T3pu} pu")

# line per unit values (given)
R_Ln1, X_Ln1 = 0.0376, 0.5277
R_Ln2, X_Ln2 = 0.4576, 1.0780

len1 = 100 # miles
len2 = 2000/5280 # ft to miles
R_Ln1pu = (R_Ln1*len1)/Z_base2
X_Ln1pu = (X_Ln1*len1)/Z_base2
Z_Ln1pu = (R_Ln1 + 1j*X_Ln1)*len1/Z_base2
print(f"Zpu_Line1 = {round(R_Ln1pu,3)} + j{round(X_Ln1pu,3)} pu")

R_Ln2pu = (R_Ln2*len2)/Z_base4
X_Ln2pu = (X_Ln2*len2)/Z_base4
Z_Ln2pu = (R_Ln2 + 1j*X_Ln2)*len2/Z_base4
print(f"Zpu_Line2 = {round(R_Ln2pu,3)} + j{round(X_Ln2pu,3)} pu")

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
print(f"Zpu_Load1 = {round(re_Ld1,3)} + j{round(im_Ld1,3)} pu")

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
print(f"Zpu_Load2 = {round(re_Ld2,3)} + j{round(im_Ld2,3)} pu\n")


# ------------------ 3 & 4 ---------------------------
# assume 4kV operating voltage at Load 2
print("3 & 4")
# Bus 6
V6_actual = 4e3
V6_pu = V6_actual/V_base6
print("per unit voltages:")
print(f"V6_pu = {round(V6_pu,3)} < 0 pu")

I6_pu = V6_pu/Z_Ld2pu

# other buses
V5_pu = V6_pu + I6_pu * x_T3pu
print(f"V5_pu = {round(np.abs(V5_pu),3)} < {round(np.angle(V5_pu),3)} pu")
V4_pu = V5_pu + I6_pu * Z_Ln2pu
print(f"V4_pu = {round(np.abs(V4_pu),3)} < {round(np.angle(V4_pu),3)} pu")
I_Ld1 = V4_pu/Z_Ld1pu
I3_pu = I6_pu + I_Ld1
V3_pu = V4_pu + I3_pu * x_T2pu
print(f"V3_pu = {round(np.abs(V3_pu),3)} < {round(np.angle(V3_pu),3)} pu")
V2_pu = V3_pu + I3_pu * Z_Ln1pu
print(f"V2_pu = {round(np.abs(V2_pu),3)} < {round(np.angle(V2_pu),3)} pu")
V1_pu = V2_pu + I3_pu * x_T1pu
print(f"V1_pu = {round(np.abs(V1_pu),3)} < {round(np.angle(V1_pu),3)} pu")

print("per unit currents:")
print(f"I6_pu = {round(np.abs(I6_pu),3)} < {round(np.angle(I6_pu),3)} pu")
print(f"I5_pu = {round(np.abs(I6_pu),3)} < {round(np.angle(I6_pu),3)}")
print(f"I4_pu = {round(np.abs(I6_pu),3)} < {round(np.angle(I6_pu),3)}")
print(f"I3_pu = {round(np.abs(I3_pu),3)} < {round(np.angle(I3_pu),3)}")
print(f"I2_pu = {round(np.abs(I3_pu),3)} < {round(np.angle(I3_pu),3)}")
print(f"I1_pu = {round(np.abs(I3_pu),3)} < {round(np.angle(I3_pu),3)}")

# calculate actual voltages
V6_actual = 4.16e3 * V6_pu
print(f"V6 = {round(np.abs(V6_actual),3)} < {round(np.angle(V6_actual),3)} pu")
V5_actual = 12.47e3 * V5_pu
print(f"V5 = {round(np.abs(V5_actual),3)} < {round(np.angle(V5_actual),3)} pu")
V4_actual = 12.47e3 * V4_pu
print(f"V4 = {round(np.abs(V4_actual),3)} < {round(np.angle(V4_actual),3)} pu")
V3_actual = 230e3 * V3_pu
print(f"V3 = {round(np.abs(V3_actual),3)} < {round(np.angle(V3_actual),3)} pu")
V2_actual = 230e3 * V2_pu
print(f"V2 = {round(np.abs(V2_actual),3)} < {round(np.angle(V2_actual),3)} pu")
V1_actual = 20e3 * V1_pu
print(f"V1 = {round(np.abs(V1_actual),3)} < {round(np.angle(V1_actual),3)} pu")

# calculate currents






# ------------------ 5 ---------------------------
# employ a pf correction solution




# ------------------ 6 ---------------------------
# reflection on pf correction


# ------------------ 7 ---------------------------
# changing base values
