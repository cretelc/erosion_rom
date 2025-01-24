import numpy as np
from uncertainties import ufloat
from uncertainties.umath import exp as uexp
from scipy.constants import pi 
from scipy.constants import elementary_charge as q
from scipy.constants import electron_mass as me
from time import perf_counter
import matplotlib.pyplot as plt

# Necessary physics equations
def vth(Te, M = me):
    ''' Calculates average thermal speed for electrons '''
    return (8*q*Te / (pi*M))**(1/2)

# Breaking down F(x) = C(g(x)/h(x)) - B 
def g(x):
    return uexp(x) - 1

def h(x):
    return x

def dg(x):
    return uexp(x)
def dh(x):
    return 1

def Fplot(x):
    return g(x)*h(x)

def F(x, *args):
    ''' 
    C = args[0]
    B = args[1] 
    '''
    return args[0] * (g(x)/h(x)) - args[1]
    #return  (g(x)/h(x)) - (args[1]/args[0])


def DF(x, *args):# C=1e-5):
    C = args[0] 
    return C * ( dg(x)*h(x) - g(x)*dh(x) ) / (h(x)**2)
    #return ( dg(x)*h(x) - g(x)*dh(x) ) / (h(x)**2)

def newton(x0, f, Df, error, maxiter, *args):
    ''' calculates the solution to f via Newtone Method'''
    xn = x0
    C, B = args
    for i in range(0, maxiter):
        fxn  = f(xn, C, B)
        Dfxn = Df(xn, C)
        #print(xn)
        if abs(fxn) < error:
            #print(f"Solution, {xn}, found after {i} iterations.")
            return xn
        if Dfxn == 0:
            #print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn / Dfxn
    #print(f'Hit max iterations without finding solution: {xn}')
    return None


def find_Vebs(Va: float, ra: float, Te: float, Vsp: float, Vbp: float, nbp: float, Ilimit:float , override_init=True):
    #for_kwargs = {'ra': ra, 'Te':Te, 'phi_sp': Vsp, 'phi_bp': Vbp, 'nbp': nbp, 'Iebs_limit': Ilimit}

    cbar    = vth(Te)    # might be one more 0  
    k1      = 2 * pi * ra**2 * q 
    k2      = nbp*cbar/ 4
    k3      = uexp((Vsp - Vbp)/Te)
    C = k1*k2*k3
    #print(f"{C}, {C/Te}")
    #print(f"B={Ilimit}")
    if override_init == True:
        Phi0 = -1
    else:
        Phi0 = (Va - Vsp) / Te
    Phi = newton(-1, F, DF, 1e-11, 100, C, Ilimit)
    Vebs = Phi * Te + Vsp
    return Vebs

def plot_phim(r, Va, phi_sp, ra):
    a = (Va - phi_sp) / (ra**2)
    c = phi_sp 
    print(f"a={a}, c={c}")
    phi_r = a*r**2 + c

    plt.plot(r*1000, phi_r)
    plt.xlabel(r'$r [mm]$')
    plt.ylabel(r'$\phi_m(r) [V]$')
    plt.xlim([0, 0.2])
    plt.grid(which='both')
    plt.show()
    return 

def main():
    Va     = -180                   # V
    Vsp    = ufloat(-17, 8)         # V
    Vbp    = ufloat(2.5, 2)         # V
    ra     = 0.57e-3                # m
    Te     =  2                     # eV
    nbp    = 1e16                   # m-3
    Ib = 0.27e-4 # A
    Rebs = 0.001
    cbar   = vth(Te)    # might be one more 0
    #print(f"cbar = {cbar:0.1f} m/s")    
    Rebs = 0.001
    #print(f"{Ib*Rebs:0.2e}")
    Vebs = find_Vebs(Va, ra, Te, Vsp, Vbp, nbp, Ib*Rebs)
    print(f"Predicted Vebs = {Vebs}")
    return 

if __name__ == "__main__":
    t0 = perf_counter()
    main()
    #ra = 0.57e-3
    #r = np.linspace(0, ra, 100)
    #plot_phim(r, -180, 20, ra)
    #print(f"Duration: {perf_counter() - t0:0.5f}s")