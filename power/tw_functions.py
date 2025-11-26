import numpy as np

V_rated = 500 # kV
pf = 0.98
theta = -np.arccos(pf)
V_r = V_rated/np.sqrt(3)

def percent_error(approx,actual):
    return np.abs(np.abs(approx) - np.abs(actual))/np.abs(actual)

def voltage_reg(Zc, Aeq, percent, times):
    SIL = V_rated**2/abs(Zc)
    P_r = times * SIL
    S_r = P_r * (pf + 1j*np.sin(theta))
    I_r = np.conj(S_r/V_r)

    Vnl = abs(Aeq*V_r)
    Vfl = abs(V_r)
    VR = (Vnl - Vfl)/Vfl * 100
    print(f"voltage regulation with {percent}% capacitive power compensation and Pr = {times}*SIL: {VR:.3f}%")


def cap_power_compensation(Z,Y,Zc):
    # capacitive power compensation
    # 25%
    print("\nABCD parameters after capacitive power compensation")
    NL = [25,50,75]

    cap25 = []
    Zeq = Z*(1-NL[0]/100)
    Yeq = Y
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    cap25.append(Aeq)
    cap25.append(Beq)
    cap25.append(Ceq)
    cap25.append(Deq)

    print("for 25% capacitive power compensation:")
    print(f'| {abs(cap25[0]):.3f} {abs(cap25[1]):.3f} |')
    print(f'| {abs(cap25[2]):.3f} {abs(cap25[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,25,1.33)

    # 50%
    cap50 = []
    Zeq = Z*(1-NL[1]/100)
    Yeq = Y
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    cap50.append(Aeq)
    cap50.append(Beq)
    cap50.append(Ceq)
    cap50.append(Deq)

    print("\nfor 50% capacitive power compensation:")
    print(f'| {abs(cap50[0]):.3f} {abs(cap50[1]):.3f} |')
    print(f'| {abs(cap50[2]):.3f} {abs(cap50[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,50,1.33)


    # 75%
    cap75 = []
    Zeq = Z*(1-NL[2]/100)
    Yeq = Y
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    cap75.append(Aeq)
    cap75.append(Beq)
    cap75.append(Ceq)
    cap75.append(Deq)

    print("\nfor 75% capacitive power compensation:")
    print(f'| {abs(cap75[0]):.3f} {abs(cap75[1]):.3f} |')
    print(f'| {abs(cap75[2]):.3f} {abs(cap75[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,75,1.33)



def react_power_compensation(Z,Y,Zc):
    # capacitive power compensation
    # 25%
    print("\nABCD parameters after reactive power compensation")
    NL = [25,50,75]

    react25 = []
    Yeq = Y*(1-NL[0]/100)
    Zeq = Z
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    react25.append(Aeq)
    react25.append(Beq)
    react25.append(Ceq)
    react25.append(Deq)

    print("for 25% reactive power compensation:")
    print(f'| {abs(react25[0]):.3f} {abs(react25[1]):.3f} |')
    print(f'| {abs(react25[2]):.3f} {abs(react25[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,25,0.5)


    # 50%
    react50 = []
    Yeq = Y*(1-NL[1]/100)
    Zeq = Z
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    react50.append(Aeq)
    react50.append(Beq)
    react50.append(Ceq)
    react50.append(Deq)

    print("\nfor 50% reactive power compensation:")
    print(f'| {abs(react50[0]):.3f} {abs(react50[1]):.3f} |')
    print(f'| {abs(react50[2]):.3f} {abs(react50[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,50,0.5)


    # 75%
    react75 = []
    Yeq = Y*(1-NL[2]/100)
    Zeq = Z
    Aeq = Deq = 1 + (Yeq*Zeq)/2
    Beq = Zeq
    Ceq = (1 + (Yeq*Zeq)/4)*Yeq
    react75.append(Aeq)
    react75.append(Beq)
    react75.append(Ceq)
    react75.append(Deq)

    print("\nfor 75% reactive power compensation:")
    print(f'| {abs(react75[0]):.3f} {abs(react75[1]):.3f} |')
    print(f'| {abs(react75[2]):.3f} {abs(react75[3]):.3f} |')

    # voltage regulation
    print('')
    voltage_reg(Zc,Aeq,75,0.5)

