import numpy as np
import tw_functions 
# constants
π = np.pi
ε0 = 8.85e-12

# specs: 4 subconductors per bundle
N = 4 # subconductors
Dab = 60 # ft
Dbc = 60 # ft
Dca = 60 # ft
d = 1.5 # ft 

# drake specs
Rc = 0.1288 # Ω/mi
GMR = 0.0375 # ft
Dsl = 1.091*(GMR*d**3)**(1/4) # ft
Deq = (Dab * Dbc * Dca)**(1/3)

# resistance & reactance calculation
Ra = Rc/N # Ω/mi
La = 2e-7 * np.log(Deq/Dsl)*1609 # H/mi
Xa = 2*π*60*La # Ω/mi
Za = Ra + 1j*Xa # Ω/mi

# admittance calculation
Ca = (2*π*ε0)/(np.log(Deq/Dsl))*1609 # C/mi
Ba = 2*π*60*Ca # S/mi
Ya = 1j*Ba

# beta, Zs, gamma, & Zc calculation
beta = 1j*np.sqrt(Xa*Ba)
Zs = np.sqrt(Xa/Ba)
gamma = np.sqrt(Za*Ya)
Zc = np.sqrt(Za/Ya)


# ----------------------------- main code ---------------------------------------

# model = input('what model do you want to use?\n1: long line (250mi)\n2: medium line (100 mi)\n3: short line (10 mi)\n-> ')
model = '1'
if(model == '1'):
    # long line
    print('---------------- long line -----------------')
    l = 250 # mi
    A = np.cos(beta*l)
    D = A
    B = 1j*Zs*np.sin(beta*l)
    C = (1/1j)*(1/Zs)*np.sin(beta*l) 

    approx = []
    approx.append(A)
    approx.append(B)
    approx.append(C)
    approx.append(D)

    print('approximated long line ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°   |')

    # first principles equations to compare against
    A = np.cosh(gamma*l)
    D = A
    B = 1j*Zc*np.sinh(gamma*l)
    C = 1j*(1/Zc)*np.sinh(gamma*l)

    actual = []
    actual.append(A)
    actual.append(B)
    actual.append(C)
    actual.append(D)

    print('\nactual ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°  |')

    # errors in ABCD parameters
    errors = []
    error = tw_functions.percent_error(approx[0],actual[0])
    errors.append(error)
    error = tw_functions.percent_error(approx[1],actual[1])
    errors.append(error)
    error = tw_functions.percent_error(approx[2],actual[2])
    errors.append(error)
    error = tw_functions.percent_error(approx[3],actual[3])
    errors.append(error)

    print('\nerrors in ABCD parameters:')
    print(f'| {errors[0]*100:.3f}% {errors[1]*100:.3f}% |')
    print(f'| {errors[2]*100:.3f}% {errors[3]*100:.3f}% |')

model = '2'
if(model == '2'):
    # medium line
    print('---------------- medium line -----------------')
    l = 100 # mi

    # Y & Z calculation
    Z = Za*l
    Y = Ya*l

    A = 1 + (Y*Z)/2
    D = A
    B = Z
    C = (1 + (Y*Z)/4)*Y 

    approx = []
    approx.append(A)
    approx.append(B)
    approx.append(C)
    approx.append(D)

    print('approximated medium line ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°   |')

    # first principles equations to compare against
    A = np.cosh(gamma*l)
    D = A
    B = 1j*Zc*np.sinh(gamma*l)
    C = 1j*(1/Zc)*np.sinh(gamma*l)

    actual = []
    actual.append(A)
    actual.append(B)
    actual.append(C)
    actual.append(D)

    print('\nactual ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°  |')

    # errors in ABCD parameters
    errors = []
    error = tw_functions.percent_error(approx[0],actual[0])
    errors.append(error)
    error = tw_functions.percent_error(approx[1],actual[1])
    errors.append(error)
    error = tw_functions.percent_error(approx[2],actual[2])
    errors.append(error)
    error = tw_functions.percent_error(approx[3],actual[3])
    errors.append(error)

    print('\nerrors in ABCD parameters:')
    print(f'| {errors[0]*100:.3f}% {errors[1]*100:.3f}% |')
    print(f'| {errors[2]*100:.3f}% {errors[3]*100:.3f}% |')

model = '3'
if(model == '3'):
    # short line
    print('---------------- short line -----------------')
    l = 10 # mi

    # Y & Z calculation
    Z = Za*l
    Y = Ya*l

    A = 1
    D = 1
    B = Z
    C = 0

    approx = []
    approx.append(A)
    approx.append(B)
    approx.append(C)
    approx.append(D)

    print('approximated short line ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°   |')

    # first principles equations to compare against
    A = np.cosh(gamma*l)
    D = A
    B = 1j*Zc*np.sinh(gamma*l)
    C = 1j*(1/Zc)*np.sinh(gamma*l)

    actual = []
    actual.append(A)
    actual.append(B)
    actual.append(C)
    actual.append(D)

    print('\nactual ABCD parameters:')
    print(f'| {np.abs(A):.3f} <{round(np.angle(A)):.3f}° {np.abs(B):.3f} <{np.angle(B):.3f}° |')
    print(f'| {np.abs(C):.3f} <{np.angle(C):.3f}° {np.abs(D):.3f} <{round(np.angle(D)):.3f}°  |')

    # errors in ABCD parameters
    errors = []
    error = tw_functions.percent_error(approx[0],actual[0])
    errors.append(error)
    error = tw_functions.percent_error(approx[1],actual[1])
    errors.append(error)
    # handle C parameter differently
    if(np.abs(approx[2]) < 1e-10):
        error = np.abs(actual[2])
    else:
        error = tw_functions.percent_error(approx[2],actual[2])
    errors.append(error)

    error = tw_functions.percent_error(approx[3],actual[3])
    errors.append(error)

    print('\nerrors in ABCD parameters:')
    print(f'| {errors[0]*100:.3f}% {errors[1]*100:.3f}% |')
    print(f'| {errors[2]*100:.3f}% {errors[3]*100:.3f}% |')



