import matplotlib.pyplot as plt 
import numpy as np
from scipy.constants import elementary_charge as q
from uncertainties import ufloat
from uncertainties.umath import log as ulog
from uncertainties.umath import sqrt as usqrt
from scipy.constants import physical_constants


aB = physical_constants['Bohr radius'][0] * (1e10)
#print(f"Bohr radius: {aB}")
#print(f"Elementary Charge: {q}")
eps0 = 8.8541878188e-12
#eps0 = 55.26349406 #e6

def xe_xsections(E, charge = "+", include_uncertainty = False):
    '''  '''
    coeffs = {
        "+"  : [ufloat(87.3, 0.9), ufloat(13.6, 0.6)],
        "++" : [ufloat(45.7, 1.9), ufloat(8.9, 1.2)]
    }
    if include_uncertainty:
        A = coeffs[charge][0]
        B = coeffs[charge][1]
    else:
        A = coeffs[charge][0].nominal_value
        B = coeffs[charge][1].nominal_value
    return A - B * np.log10(E)

def nuclear_stopping_pwr(eps):
    ''' Nuclear stopping power based on the Krypton-Carbon potential '''
    return 0.5*ulog(1 + 1.2288*eps) / (eps + 0.1728*usqrt(eps) + 0.008*(eps**(0.1504)))

def Lindhard_screen_length(Zi, Zs):
    ''' Lindhard screen length [Eqn. 4, Eckstein & Preuss, 2003]
    verified
    Keyword arguments 
    Zi -- incident particle atomic number
    Zs -- target particle atomic number 

    Returns 
    aL -- Lindhard screen length, m
    '''
    aL  = aB * (9*(np.pi**2) / 128)**(1/3) * (Zi**(2/3) + Zs**(2/3))**(-0.5)
    #print(f"Lindhard Screen Length: {aL:0.3e}")
    return aL

def reduced_energy(E, Mi, Ms, Zi, Zs):
    ''' Calculate reduced energy for the nuclear stopping power.
    
    Keyword arguments 
    E  -- incident particle energy, ??
    Mi -- incident particle atomic mass, AMU
    Ms -- target particle atomic mass, AMU
    Zi -- incident particle atomic number
    Zs -- target particle atomic numbers 

    Returns 
    re -- reduced energy, ??
    '''
    aL = Lindhard_screen_length(Zi, Zs)
    eps0_local = 1.42e-40
    C = (Mi / (Ms + Mi)) * (aL / (Zi*Zs)) * (4*np.pi*eps0_local/(q**2)) # the q*q term is ~1e-38
    re = C * E
    return re

def Eckstein_yield_model(E, params):
    ''' Calculates Xe-moly sputter yield using the Eckstein model [Yim, 2017]
    - I may need to used the Yim version
    
    Keyword arguments:
    E   -- incident particle energy,
    params -- model parameters and variables 
        Q   -- fit parameter
        lam -- fit parameter
        Eth -- sputtering threshold energy, fit parameter
        sn  -- nuclear stopping power
        re -- reduced energy
    '''
    Q, lam, mu, Eth, sn, re = params
    X1 = (E/Eth - 1) ** mu
    w = re + 0.1728*re**(1/2) + 0.008*re**(0.1504) # Key to Yim's formulation
    return Q * sn * (X1 / ((lam/w) + X1))


def calculate_yield_Xe_Mo(E: np.ndarray):
    target = 'Mo'
    incident = 'Xe'
    Zs = 42         # atomic number
    Zi = 54        # atomic number
    Ms = 95.95    # g/mol, atomic weight
    Mi = 131.29    # g/mol, atomic weight

    # Eckstein parameters -> get from csv
    Q   = 18.4
    mu  = 2.2
    lam = 1.7
    Eth = 14.2 # eV, threshold energy
    
    aL   = Lindhard_screen_length(Zi, Zs)
    re   = reduced_energy(E, Mi, Ms, Zi, Zs)
    sn   = nuclear_stopping_pwr(re)
    P = [Q, lam, mu, Eth, sn, re]
    Y = Eckstein_yield_model(E, P)
    return Y

def calculate_yield_Xe_C(E: np.ndarray):
    target = 'C'
    incident = 'Xe'
    Zs = 6        # target atomic number
    Zi = 54        # incident atomic number
    Ms = 95.95     # g/mol, atomic weight
    Mi = 131.29    # g/mol, atomic weight

    # Eckstein parameters -> get from csv
    Q   = 18.4
    mu  = 2.2
    lam = 1.7
    Eth = 14.2 # eV, threshold energy
    
    aL   = Lindhard_screen_length(Zi, Zs)
    re   = reduced_energy(E, Mi, Ms, Zi, Zs)
    sn   = nuclear_stopping_pwr(re)
    P = [Q, lam, mu, Eth, sn, re]
    Y = Eckstein_yield_model(E, P)
    return Y

def calculate_yield(E, Ztarget, Zincident, Mtarget, Mincident, params):
    target = 'C'
    incident = 'Xe'
    Q, mu, lam, Eth = params
    atomic_numbers = {'Xe':54,
                      'Kr':36,
                      'Ar':18,
                      'N' :7, 
                      'Mo':42,
                      'C': 6}
    atomic_weights = {'Xe':131.29,
                      'Kr':83.80,
                      'Ar':39.95,
                      'N' :14.007, 
                      'Mo':95.95,
                      'C': 12.011}
    #Zs = atomic_numbers[target_symbol]      # target atomic number
    #Zi = atomic_numbers[incident_symbol]    # incident atomic number
    #Ms = atomic_weights[target_symbol]     # g/mol, atomic weight
    #Mi = atomic_weights[incident_symbol]    # g/mol, atomic weight

    # Eckstein parameters -> get from csv
    #Q   = 18.4
    #mu  = 2.2
    #lam = 1.7
    #Eth = 14.2 # eV, threshold energy
    
    aL   = Lindhard_screen_length(Zincident, Ztarget)
    re   = reduced_energy(E, Mincident, Mtarget, Zincident, Ztarget)
    sn   = nuclear_stopping_pwr(re)
    P = [Q, lam, mu, Eth, sn, re]
    Y = Eckstein_yield_model(E, P)
    return Y





if __name__ == "__main__":
    E = np.linspace(15, 1000, 500)
    Y = calculate_yield_Xe_Mo(E)
    plt.loglog(E, Y)
    plt.xlabel('Xenon ion energy [EV]')
    plt.ylabel('Sputter yield [atoms/ion]')
    plt.grid(which='both')
    plt.xlim([10, 1000])
    plt.ylim([0.0000001,3])
    plt.show()
