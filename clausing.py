import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt

def big_C(c1, c2):
    return (1/(c1) + 1/(c2))**(-1)

def little_c(M, d, l):
    return 3.8 * np.sqrt(300/M) * ((d**3) / l)

def c_apert(M, r):
    return 3.64 * np.sqrt(300/M) * np.pi * r**2

def calculateClausingFactor(M, ds, ts, da, ta, lg):
    ra = da/2
    c1 = little_c(M, ds, ts+lg)
    c2 = little_c(M, da, ta)
    C = big_C(c1, c2)
    capert = c_apert(M, ra)
    Clausing = capert/C
    return Clausing

def simpleClausingFactor(ra, s):
    return 4 * (ra/s)**2

def contour_test():
    ts = 0.38
    ra_arr = np.linspace(0.55, 1)
    ra = 0.55
    da = 2*ra
    rs = 3.415
    ds = 2*rs
    ta = 2.73
    ta_arr = np.linspace(0.05, 0.55)
    lg = 2.36
    M  = 131.3

    fig, ax1 = plt.subplots()
    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Clausing vs ta
    c1 = little_c(M, 2*rs, ts+lg)
    c2 = little_c(M, 2*ra, ta_arr)
    C = big_C(c1, c2)
    capert = c_apert(M, ra)
    Clausing = capert/C
    # Plot the first dataset on the left y-axis
    ax1.plot(Clausing, ta_arr, color='blue')
    ax1.set_ylabel(r'$t_a$', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Clausing vs ra
    c1 = little_c(M, 2*rs, ts+lg)
    c2 = little_c(M, 2*ra_arr, ta)
    C = big_C(c1, c2)
    capert = c_apert(M, ra_arr)
    Clausing = capert/C
    # Plot the first dataset on the left y-axis
    ax2.plot(Clausing, ra_arr, color='red')
    ax2.set_ylabel(r'$r_a$', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_xlabel('Clausing Factor')
    plt.show()

def main():
    ts = 0.15
    ra = 2.05
    da = 2*ra
    rs = 3.415
    ds = 2*rs
    ta = 2.73
    lg = 2.36
    M  = 131.3

    clausing_factor = calculateClausingFactor(M, ds, ts, da, ta, lg)
    print(f"Clausing factor: {clausing_factor:0.3f}")
    return


if __name__ == '__main__':
    contour_test()
