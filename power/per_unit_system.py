import numpy as np

# ------------------ 1 & 2 -------------------------
# set the base values (given)
S_base = 100 * np.power(10,6) # 100MVA
print(f"S_base = {round(S_base*1/np.power(10,6))} MVA")
V_base1 = 20 * np.power(10,3) # 20kV
print(f"V_base1 = {round(V_base1*0.001)} kV")
V_base2 = 230 * np.power(10,3) # 230kV
print(f"V_base2 = {round(V_base2*0.001)} kV")
V_base4 = 12.47 * np.power(10,3) # 12.47kV
print(f"V_base4 = {V_base4*0.001} kV")
V_base6 = 4.16 * np.power(10,3) # 4.16kV
print(f"V_base6 = {V_base6*0.001} kV")

Z_base2 = np.power(V_base2,2)/(S_base)
print(f"Z_base2 = {round(Z_base2)} ohms")
Z_base4 = np.power(V_base4,2)/(S_base)
print(f"Z_base4 = {round(Z_base4,3)} ohms")
Z_base6 = np.power(V_base6,2)/(S_base)
print(f"Z_base6 = {round(Z_base6,3)} ohms")

# transformer per unit values (given: 0.1pu)
x_T1pu = 0.1*(S_base/(100 * np.power(10,6)))
print(f"Xpu_T1 = {x_T1pu} pu")
x_T2pu = 0.1*(S_base/(75 * np.power(10,6)))
print(f"Xpu_T2 = {round(x_T2pu,2)} pu")
x_T3pu = 0.1*(S_base/(25 * np.power(10,6)))
print(f"Xpu_T3 = {x_T3pu} pu")

# line per unit values (given)
R_Ln1, X_Ln1 = 0.0376, 0.5277
R_Ln2, X_Ln2 = 0.4576, 1.0780

Z_Ln1pu = R_Ln1 + 1j*X_Ln1/Z_base2
print(f"Zpu_Line1 = {R_Ln1} + j{X_Ln1} pu")
Z_Ln2pu = R_Ln2 + 1j*X_Ln2/Z_base4
print(f"Zpu_Line2 = {R_Ln2} + j{X_Ln2} pu")

# load per unit values Z_Load = V^2/S < (cos^-1(pf))

# Load 1 
# given values
S_Ld1 = 50 * np.power(10,6) # 50MVA
pf_Ld1 = 0.85

# find load 1 per unit impedance
phase_Ld1 = np.arccos(pf_Ld1)
re_Ld1 = np.power(V_base4,2)/(S_Ld1) * np.cos(phase_Ld1)
im_Ld1 = np.power(V_base4,2)/(S_Ld1) * np.sin(phase_Ld1)
Z_Ld1 = re_Ld1 + 1j*im_Ld1
Z_Ld1pu = Z_Ld1/Z_base4

# Load 2
# given values
S_Ld2 = 20 * np.power(10,6) # 20MVA
pf_Ld2 = 0.9

# find load 2 per unit impedance
phase_Ld2 = np.arccos(pf_Ld2)
re_Ld2 = np.power(V_base6,2)/S_Ld2 * np.cos(phase_Ld2)
im_Ld2 = np.power(V_base6,2)/S_Ld2 * np.sin(phase_Ld2)
Z_Ld2 = re_Ld2 + 1j*im_Ld2
Z_Ld2pu = Z_Ld2/Z_base6


# ------------------ 3 ---------------------------
# assume 4kV operating voltage at Load 2
