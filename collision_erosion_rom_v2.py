## Dependencies
# JSON
import json
# Numpy
import numpy as np
from numpy import pi
# Matplotlib
import matplotlib.pyplot as plt
# Scipy Constants
from scipy.constants import Avogadro as N_A
from scipy.constants import elementary_charge as q
from scipy.constants import Boltzmann as kB
from scipy.constants import epsilon_0 as eps0
from scipy.constants import electron_mass as me
# Scipy Optimize
from scipy.optimize import curve_fit
# Pandas
import pandas as pd
# Time
from time import perf_counter
# Uncertainties
from uncertainties import ufloat
from uncertainties.umath import log as ulog
from uncertainties.umath import log10 as ulog10
# My libraries
from clausing import simpleClausingFactor
import wgw_model as wgw
import eckstein_yield as ey
from myEBS_library import find_Vebs


# Tests
def vol_test():
    r = 1
    R = 1
    h = 3
    #print(f"{beam_volume_trap(h, r, r): 0.3f}")
    #print(f"{plume_volume(r, h, 0):0.3f}")
    V1 = beam_volume_trap(h, r, r)
    V2 = plume_volume(r, h, 0)
    return V1==V2

# Conversions
def AMU2kg(AMU):
    M = (1e-3)*AMU / N_A
    return M

# Beam and Plume Volume Functions Carbon Backsputtering Functions
def plume_volume(Rth, n_ds = 10, div = 20):
    div_rad = div * np.pi / 180
    #print(div_rad)
    return ( (np.pi) * Rth**3 ) * ( ( n_ds**3 / 3) * np.tan( div_rad )**2 + ( n_ds**2 ) * np.tan( div_rad ) + n_ds )

def beam_volume_trap(h, r1, r2):
    ''' Calculates the beam volume between the screen and accel grid using 
        approximating the beam as a truncated trapezoid.

        returns 
        beam volume, assuming units of all parameters is the same    
    '''
    return ( pi * h / 3 ) * (r2**2 + r1*r2 + r1**2)

# Custom Models
def calculate_parameters_from_clausing(C): 
    ni = (-1.65*C + 2.228) * (1e17)
    no = (-5.281*C + 3.050) * (1e18)
    y  = (0.2562*C + 0.0203)
    Te = (1.6636*C + 6.3558)
    return ni, no, y, Te

def beam_area_model(j_bar=1, a=0.619, b=0.011, c=-0.045):
    return a*j_bar + (b/j_bar) + c

def f(x, a, b):
    return a*np.log(1+b*x)

def dfdt(x, a, b):
    return a*b/(1 + b*x)

def vth(Te, M = me):
    ''' Calculates average thermal speed for electrons '''
    return np.sqrt(8*q*Te / (pi*M))

# Electron Backstreaming -- Wirz Integration Model
def Iebs(Va, Te, ra, Vsp, Vbp, nbp, cbar):
    ''' Calculates backstreaming electron curren per Eqn. 5 in Wirz, 2011  '''
    A = (ra**2)*pi*q*Te/(Va - Vsp)
    B = nbp*cbar / 4
    C = np.exp((Vsp - Vbp)/Te)
    D = np.exp((Va - Vsp)/Te) - 1
    return A*B*C*D

def potential_ratio(r1, r2):
    rmin = min(r1, r2)
    rmax = max(r1, r2)
    dr = rmax = rmin
    return r1/(r1+r2), r2/(r1+r2)
# - - - 

# Cross sections
def xe_xsections(E, charge = "+", uncertain=False):
    ''' Calculates the xenon CEX collision cross sections [Miller, 2002]

    Arguments 
    E      -- incident ion energy
    charge -- incident ion charge (default = singly charged) 
    '''
    ucoeffs = {
        "+"  : [ufloat(87.3, 0.9), ufloat(13.6, 0.6)],
        "++" : [ufloat(45.7, 1.9), ufloat(8.9, 1.2)]
    } # save ufloat for later when I have more time

    coeffs = {
        "+"  : [87.3, 13.6],
        "++" : [45.7, 8.9]
    } 
    A = coeffs[charge][0]
    B = coeffs[charge][1]
    #if uncertain:
    #    return A - B * ulog10(E)
    #else:
    #    return A - B * ulog10(E)
    return A-B*np.log10(E)
# - - - 

# Carbon redeposition 
def redep_flux_Xe_C(Ib, Rth, E):
    Y   = ey.calculate_yield(E, 6, 54, 12.011, 131.29, [4, 0.8, 1.8, 21])
    Ath = np.pi * Rth**2
    carbon_flux = Y*Ib / (Ath * q)
    return carbon_flux

def sputterant_redep_rate(IB, Rch, Rth, Ns, Ni, Ms, Mi, rho_s, E, params = [4, 0.8, 1.8, 21], sc=1):
    ''' Returns the rate that carbon sputterants deposit back on the thruster '''
    Y = ey.calculate_yield(E, Ns, Ni, Ms, Mi, params)
    Ath = np.pi * Rth * Rth
    Ach = np.pi * Rch * Rch
    s_avg = (Y * IB / q) * (Ath / Ach)
    return s_avg

def carbon_surface_coverage(sdot, Icex, Y=0.99):
    ''' Calculates carbon surface coverage '''
    Y_C_Mo = Y
    thetaC = sdot / (Icex * Y_C_Mo)
    return thetaC


# Charge Exchange Generation
def cex_generation_rate(j, n0, single_xsection, volume):
    ''' Returns cex ions generated per second (ions/s) '''
    return (j/q) * n0 * single_xsection * volume

def cex_flux(j, n0, single_xsection, volume):
    ''' Returns the flux of CEX ions generated (ions / m2 s)'''
    return (j/q) * n0 * single_xsection * volume
# - - -

# Parameter Erosion Models
def accel_radius_erosion(dt, hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
    # Pull arguments
    j_bar, y, P1, P2, eta_cex, n_ds, phi_p    = hyperparams
    Rth, div                                        = thruster
    rs, ts, ra1, ra2, ta, lg, s, M_grid, rho_grid         = grids
    ni_dis, n0_dis, Te_dis                          = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C                 = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi               = operating_conditions

    '''
    j_bar, y                    = hyperparams
    rs, ra, lg, s               = grid_geometry
    M_grid, rho_grid            = grid_mats
    n0, Ec, E_cex, Vd, Va, M    = model_inputs
    '''
    # convert special variables to their preferred units
    lg_m   = lg * (1e-3)        # mm -> m
    n0_mm3  = n0_dis * (1e-9)   # m-3 -> mm-3

    erosion_control = np.array([P1, P2])
    ra              = np.array([ra1, ra2])

    jCL              = j_CL(Mi, Vd-Va, lg_m)                                # -> A m-2 this is overpredicting Ib but it's ok
    jb               = j_bar * jCL                                          # j_bar * jBohm
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)                        # sq.angstrom -> m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)                       # sq.angstrom -> m2
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)                      # -> atoms/ion; produces reasonable results
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)                    # -> atoms/ion; produces reasonable results
    rb               = min(ra) * beam_area_model(j_bar)**(1/2)              # -> mm
    Vbeam            = beam_volume_trap(lg, rs, rb)                         # -> mm3
    Gamma_CEX        = cex_generation_rate(jb, n0_mm3, sigma_p, Vbeam)      # -> ions/s, Gamma^C_CEX 
    sputter_flux     = erosion_control * cex_erosion_flux(Gamma_CEX, y, 
                                                          Y_p, Y_pp, 
                                                          sigma_p, sigma_pp) # atoms/s
    
    dra_dt_0         = sputter_flux * M_grid / (N_A * rho_grid) # mm/s
    # Convert to mm/khr before returning value
    dra_dt           = dra_dt_0*(60)*(60)*(1000)                # mm / khr
    r_erode          = dt*dra_dt + ra
    return r_erode

def accel_thickness_erosion(dt,  hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
    ''' Calculates rate of carbon backsputter to the thruster

        arguments (units)
        j_bar --- ()
        y     --- ()
        Rth   --- (m)
        rs, ts, ra, ta, lg, s --- (mm)
        M_grid    --- (kg)
        rho_grid  --- (g/mm2)
        n0        --- (m-3)
        Ec, E_cex --- (eV)
        Vd, Va    --- (V)
        M         --- (AMU, g/mol)
    '''
        # Pull arguments
    j_bar, y, _, _, eta_cex, n_ds, phi_p    = hyperparams
    rs, ts, _, _, ta, lg, s, M_grid, rho_grid         = grids

    # Pull arguments 
    #j_bar, Ibar, y, eta_cex, n_ds, _        = hyperparams
    Rth, diverg                             = thruster
    #rs, ts, ra, ta, lg, s, M_grid, rho_grid = grids
    ni_dis, n0_dis, Te_dis                  = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C         = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi       = operating_conditions
    
    # conversions where necessary
    rho_grid_m = rho_grid * (1e9) # g/mm3 -> g/m3
    Ath  = np.pi * Rth**2
    javg = IB / Ath
    

    # Same as accel_radius_erosion_model()
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)    # m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)   # m2

    # updated for accel grid thickness ---
    
    Vplume           = plume_volume(Rth, n_ds=n_ds, div = diverg)                       # plume volume, m3  
    s_avg            = sputterant_redep_rate(IB, Rch, Rth, Ns, Ni, Ms, Mi, rho_C, Vd)   # redeposition rate of facility sputterants -> atoms/s
    r_CEX            = eta_cex*cex_generation_rate(IB, n0_fac, sigma_p, Vplume)         # rate of cex ions being generated -> ions/s
    G_CEX            = eta_cex*cex_generation_rate(javg, n0_fac, sigma_p, Vplume)       # flux of cex ions being generated -> ions/m2 s
    theta_C          = carbon_surface_coverage(s_avg, r_CEX)                            # surface coverage of carbon
    Y_p              = ey.calculate_yield_Xe_Mo(abs(Va))                                # -> atoms/ion; produces reasonable results
    Y_pp             = ey.calculate_yield_Xe_Mo(2*abs(Va))                              # -> atoms/ion; produces reasonable results
    sputter_flux     = cex_erosion_flux(G_CEX, y, Y_p, Y_pp, 
                                        sigma_p, sigma_pp, 
                                        carbon_surface_coverage=theta_C)                # atoms/s

    # This equation needs sputter flux to actually be a flux...
    dta_dt           = sputter_flux * M_grid / (N_A * rho_grid_m) 
    dta_dt           = dta_dt*1000*(60)*(60)*(1000) # mm / khr
    t_erode          = ta - dt * dta_dt 
    return t_erode
    
def screen_thickness_erosion(dt, hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
    
    # Pull arguments
    j_bar, y, _, _, eta_cex, n_ds, phi_p    = hyperparams
    rs, ts, _, _, ta, lg, s, M_grid, rho_grid     = grids
    # Pull arguments
    Rth, div                                = thruster
    ni_dis, n0_dis, Te_dis                  = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C         = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi       = operating_conditions

    # convert special variables to their preferred units
    lg_kg      = lg * (1e-3)
    n0_mm3     = n0_dis * (1e-9)
    rho_grid_m = rho_grid * (1e9) # g/mm3 -> g/m3

    jBohm        = bohm_current_density(ni_dis, Te_dis, Mi)             # A m-2
    Y_p          = ey.calculate_yield_Xe_Mo(phi_p)                      # atoms/ion           
    Y_pp         = ey.calculate_yield_Xe_Mo(2*phi_p)                    # atoms/ion
    sputter_flux = double_single_sputter_yield(jBohm, y, Y_p, Y_pp)     # atoms m-2 s-1
    dts_dt_0     = sputter_flux * M_grid / (N_A * rho_grid_m)           # m/s
    dts_dt       = dts_dt_0 * 1000 * 60 * 60 * 1000                     # mm / khr
    ts_erode     = ts - dt * dts_dt 
    return ts_erode
# -----------------

def grab_data():
    measured_da_data = pd.read_csv('wirz-time_dependent_fig11.csv', names=['T', 'da'])
    xdata      = measured_da_data['T'].to_numpy()
    ydata      = measured_da_data['da'].to_numpy() / 2 # da -> ra
    return xdata, ydata

def grab_data_general(csv_file, names):
    measured_da_data = pd.read_csv(csv_file, names=names)
    xdata      = measured_da_data[names[0]].to_numpy()
    ydata      = measured_da_data[names[1]].to_numpy()
    return xdata, ydata




## Functions
def erosion_rate(flux, sputter_yield, target_atomic_weight, target_density):
    return flux * sputter_yield * (target_atomic_weight / (target_density * N_A))

def cex_erosion_flux(flux_cex, single_double_ratio, single_yield, double_yield, single_xsection, double_xsection, carbon_surface_coverage=0):
    A = flux_cex * (1 - carbon_surface_coverage) / (1+single_double_ratio)
    B = (single_double_ratio/2) * (double_xsection/single_xsection)
    return A * (single_yield + B*double_yield)

# Current Densities
def bohm_current_density(n0, Te_eV, M_amu):
    Te = Te_eV * q
    M = AMU2kg(M_amu)
    return 0.606 * n0 * q * (Te/M)**0.5

def j_CL(M , V, d):
    ''' Calculates the Child-Langmuir current density
    
    Arguments
    M -- ion mass, AMU 
    V -- total potential, Vd - Va
    d -- intergrid distance
    '''
    M_kg = AMU2kg(M)
    return (4 * eps0 / 9) * (2 * q / M_kg)**(0.5) * (V**(3/2) / (d**2))


def double_single_sputter_yield(j, double_single_ion_ratio, single_yield, double_yield):
    ''' Calculates sputter yield from single and doubley charged ions, generally'''
    return ( j / ( q*(1+double_single_ion_ratio) )) * (single_yield + (double_single_ion_ratio/2)*double_yield)


def simple_erosion_model1():
    # Print Iterations? 
    print_iters = True 
    ## User set values
    # hyperparameters ---------------------------------------
    #j_bar    = 0.35     # normalized beamlet current
    eta_p    = 0.004    # cex ion impingement probability in the plume
    eta_b1   = 0.75     # upstream cex ion impingement probability between the grids
    eta_b2   = 1-eta_b1 # downstream cex ion impingement probability between grids
    n_ds     = 10       # thruster radius's downstream 
    Rebs     = 0.001    # electron backstreaming ratio limit
    w        = 0.5      # Weight of upstream accel vertice used in ra_avg calc, [0,1]
    # thruster ----------------------------------------------
    lg       = 0.36     # grid separation, mm
    rs       = (1.91)/2 # screen grid radius, mm
    ra0      = (1.14)/2 # Initial accel grid radius, mm
    s        = 2.24     # aperture center spacing, mm
    ta0      = 0.38     # accel grid thickness. mm
    Rth      = 0.15     # thruster major radius, m
    ts0      = 0.38     # screen grid thickness, mm
    M_grid   = 95.95    # grid material atomic mass, g/mol
    rho_grid = 0.01022  # grid material density, g/mm3
    diverg   = 20       # thruster divergence angle, degrees
    phi_p    = 0        # beam plasma potential, V
    ni_dis   = 1e17     # discharge ion density, m-3
    Te_dis   = 15       # discharge chamber electron temp, eV
    # facility ----------------------------------------------
    n0_fac   = 2e18     # facility number density, m-3
    Rch      = 2        # chamber radius, m
    Lch      = 3        # chamber length, m
    Ns       = 6        # chamber wall material atomic number
    Ms       = 12.011   # chamber wwall material atomic mass, g/mol
    rho_C    = 2.25e6   # redeposited material density, g/m3
    Te_beam  = 2        # beam electron temperature, eV
    nbp      = 5e16     # plasma density near accel grid, m-3
    # model inputs/Operating conditions ---------------------
    IB       = 1.76     # beam current, A
    Ib       = 2.7e-4   # beamlet current, A
    Vd       = 1100     # Discharge potential above cathode, V
    Va       = -180     # Accel potential below cathode, V
    phi_dis  = 32       # Discharge plasma potential over screen grid, V (unused)
    Ni       = 54       # propellant atomic number
    Mi       = 131.27   # propellant atomic mass, AMU
    Ec       = 400      # Energy at collision, +\-200V
    E_cex    = 400      # Energy of CEX ions colliding with grid, +\-200V
    
    leff  = wgw.eff_grid_gap(lg*(1e-3), ts0*(1e-3), 2*(rs*(1e-3)))
    As    = 2*pi*(rs*1e-3)**2 
    Ibohm = As*bohm_current_density(ni_dis, Te_dis, Mi)
    jCL = j_CL(Mi, Vd-Va, leff)
    ICL = As*jCL
    j_bar = Ib /( As * j_CL(Mi, Vd-Va, leff))
    print(f"j_bar    = {j_bar:0.2f}")
    print(f"I_cl     = {ICL :0.2e}")
    print(f"Ib       = {Ib:0.2e}")
    print(f"Ibohm    = {Ibohm:0.2e}")
    print(f"Ib/IBohm = {Ib/Ibohm:0.2f}")

    # initialize results arrays
    dt      = 0.1
    T_end   = 8.2
    tsteps  = np.arange(0, T_end+dt, dt)
    ra1_t   = np.zeros(len(tsteps))
    ra2_t   = np.zeros(len(tsteps))
    rb_t    = np.zeros(len(tsteps))
    ta_t    = np.zeros(len(tsteps))
    ts_t    = np.zeros(len(tsteps))

    VebsWirz_t = np.zeros(len(tsteps))
    y_arr  = np.zeros(len(tsteps))
    n0_arr = np.zeros(len(tsteps))
    Te_arr = np.zeros(len(tsteps))
    ni_arr = np.zeros(len(tsteps))
    # Populates results at t=0
    ra1_t[0] = ra0
    ra2_t[0] = ra0
    rb_t[0]  = ra0*(beam_area_model(j_bar=j_bar)**(1/2))
    ta_t[0]  = ta0
    ts_t[0]  = ts0
    # Electron Backstreaming model
    Vsp           = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
    dv            = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb_t[0]*(1e-3), 2*ra0*(1e-3))
    VebsWirz_t[0] = -dv+find_Vebs(Va, ra0*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )

    CF = simpleClausingFactor(ra=ra0, s=s)
    # Update discharge chamber plasma parameters at t=0
    ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
    y_arr[0]  = y
    n0_arr[0] = n0_thstr
    Te_arr[0] = Te_thstr
    ni_arr[0] = ni_thstr



    i = 1
    start_time = perf_counter()
    for t in tsteps[1:]:
        t0 = perf_counter()
        # Fill model inputs 
        hyper = (j_bar, y, eta_b1, eta_b2, eta_p, n_ds, phi_dis)
        thstr = (Rth,  diverg)
        grids = (rs, ts_t[i-1], ra1_t[i-1], ra2_t[i-1], ta_t[i-1], lg, s, M_grid, rho_grid)
        disch = (ni_thstr, n0_thstr, Te_thstr)
        fclty = (n0_fac, Rch, Lch, Ns, Ms, rho_C)
        opcon = (IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi)
   
        # Calculate new accel grid radius 
        ra_ds, ra_us = accel_radius_erosion(dt, hyper, thstr, grids, disch, fclty, opcon) 
        ta_t[i]      = accel_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ts_t[i]      = screen_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ra1_t[i] = ra_ds
        ra2_t[i] = ra_us
        # Electron Backstreaming 
        rb      = min(ra1_t[i], ra2_t[i])*(beam_area_model(j_bar=j_bar)**(1/2))
        rb_t[i] = rb
        ra_avg  = np.average([ra_us, ra_ds], weights=[w,1-w])
        Vsp     = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
        dv      = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb*(1e-3), 2*ra_avg*(1e-3))
        VebsWirz_t[i] = -dv + find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )
        # Update Clausing Factor 
        CF = simpleClausingFactor(ra=min(ra1_t[i], ra2_t[i]), s=s)
        # Update discharge chamber plasma parameters
        ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
        y_arr[i]  = y
        n0_arr[i] = n0_thstr
        Te_arr[i] = Te_thstr
        ni_arr[i] = ni_thstr
        i+=1
        if print_iters == True:
            print(f"")
            print(f"Vsp = {Vsp:0.3f}, dv = {dv:0.3f}")
            print(f'ra_min = {ra_ds}, rb = {rb}')
            print(f"Clausing Factor = {CF}")
            print(f"Iteration time: {perf_counter() - t0:0.4f}s")
        else:
            pass
    
    # Present results
    print(f"Total Runtime: {perf_counter() - start_time: 0.5f} s")
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    ax = axs[0]
    ax.plot(tsteps, ra1_t, 'k-', label=r'$r_{a,ds}$')
    ax.plot(tsteps, ra2_t, 'k--', label=r'$r_{a,up}$')
    ax.plot(tsteps, rb_t, 'r-', label=r'$r_b$')
    ax.set_ylabel(r'$[mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[1]
    ax.plot(tsteps, ta_t, 'k-', label = r"$t_a$")
    ax.plot(tsteps, ts_t, 'k--', label = r"$t_s$")
    ax.set_title('Grid thickness')
    ax.set_xlabel(r'$Time [khr]$', fontsize=18)
    ax.set_ylabel(r'$[mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()

    # Plot discharge parameters
    fs = 12
    use_color = 'k'
    xlabel = r'Time [$khr$]'
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12,8))
    ax = axs[0][0]
    ax.plot(tsteps, y_arr, color=use_color)
    ax.set_title(r'Double-single ion ratio ($\gamma$)')
    #ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('', fontsize=fs)
    ax.grid(which='both')

    ax = axs[0][1]
    ax.plot(tsteps, n0_arr, color=use_color)
    ax.set_title(r'Neutral density ($n_0$)')
    #ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
    ax.grid(which='both')

    ax = axs[1][0]
    ax.plot(tsteps, Te_arr, color=use_color)
    ax.set_title(r'Electron temperature ($T_e$)')
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$eV$', fontsize=fs)
    ax.grid(which='both')
 
    ax = axs[1][1]
    ax.plot(tsteps, ni_arr, color=use_color)
    ax.set_title(r'Ion density ($n_i$)')
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
    ax.grid(which='both')

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(tsteps, abs(VebsWirz_t), 'k')
    ax.set_xlabel(r'Time $[khr]$', fontsize=fs)
    ax.set_ylabel(r'$|V_{ebs}|$', fontsize=fs)
    ax.grid(which='both')
    plt.show()

    return tsteps, ra1_t, ra2_t, ta_t, ts_t

def json_simple_erosion_model(input_file, simulation_duration, step_size):
    # Print Iterations? 
    print_iters = True 
    # Extract json file
    with open(input_file, 'r') as openfile:
        inputs = json.load(openfile)

    print(inputs)
        # hyperparameters ---------------------------------------
    eta_p    = inputs['eta_p']['value']
    eta_b1   = inputs['eta_b1']['value']
    eta_b2   = inputs['eta_b2']['value']
    n_ds     = inputs['n_ds']['value']
    Rebs     = inputs['Rebs']['value']
    w        = inputs['w']['value']
    # thruster ----------------------------------------------
    lg       = inputs['lg']['value']
    rs       = inputs['rs']['value']
    ra0      = inputs['ra']['value']
    s        = inputs['s']['value']
    ta0      = inputs['ta']['value']
    Rth      = inputs['Rth']['value']
    ts0      = inputs['ts']['value']
    M_grid   = inputs['M_grid']['value']
    rho_grid = inputs['rho_grid']['value']
    diverg   = inputs['diverg']['value']
    phi_p    = inputs['phi_p']['value']
    ni_dis   = inputs['ni_dis']['value']
    Te_dis   = inputs['Te_dis']['value']
    # facility ----------------------------------------------
    n0_fac   = inputs['n0_fac']['value']     
    Rch      = inputs['Rch']['value']        
    Lch      = inputs['Lch']['value']
    Ns       = inputs['Ns']['value']
    Ms       = inputs['Ms']['value']
    rho_C    = inputs['rho_C']['value']
    Te_beam  = inputs['Te_beam']['value']
    nbp      = inputs['nbp']['value']
    # model inputs/Operating conditions ---------------------
    IB       = inputs['IB']['value']
    Ib       = inputs['Ib']['value']
    Vd       = inputs['Vd']['value']
    Va       = inputs['Va']['value']
    phi_dis  = inputs['phi_dis']['value']
    Ni       = inputs['Ni']['value']
    Mi       = inputs['Mi']['value']
    Ec       = inputs['Ec']['value']
    E_cex    = inputs['E_cex']['value']
    

    # Calculates Secondary variables
    leff  = wgw.eff_grid_gap(lg*(1e-3), ts0*(1e-3), 2*(rs*(1e-3)))
    As    = 2*pi*(rs*1e-3)**2 
    Ibohm = As*bohm_current_density(ni_dis, Te_dis, Mi)
    jCL = j_CL(Mi, Vd-Va, leff)
    ICL = As*jCL
    j_bar = Ib /( As * j_CL(Mi, Vd-Va, leff))
    print(f"j_bar = {j_bar:0.2f}")
    print(f"I_cl  = {ICL :0.2e}")
    print(f"Ib    = {Ib:0.2e}")
    print(f"Ibohm = {Ibohm:0.2e}")
    print(f"Ib/IBohm = {Ib/Ibohm:0.2f}")

    # initialize results arrays
    dt      = step_size
    T_end   = simulation_duration
    tsteps  = np.arange(0, T_end+dt, dt)
    ra1_t   = np.zeros(len(tsteps))
    ra2_t   = np.zeros(len(tsteps))
    rb_t    = np.zeros(len(tsteps))
    ta_t    = np.zeros(len(tsteps))
    ts_t    = np.zeros(len(tsteps))

    VebsWirz_t = np.zeros(len(tsteps))
    y_arr  = np.zeros(len(tsteps))
    n0_arr = np.zeros(len(tsteps))
    Te_arr = np.zeros(len(tsteps))
    ni_arr = np.zeros(len(tsteps))
    # Populates results at t=0
    ra1_t[0] = ra0
    ra2_t[0] = ra0
    rb_t[0]  = ra0*(beam_area_model(j_bar=j_bar)**(1/2))
    ta_t[0]  = ta0
    ts_t[0]  = ts0
    # Electron Backstreaming model
    Vsp           = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
    dv            = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb_t[0]*(1e-3), 2*ra0*(1e-3))
    VebsWirz_t[0] = -dv+find_Vebs(Va, ra0*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )

    CF = simpleClausingFactor(ra=ra0, s=s)
    # Update discharge chamber plasma parameters at t=0
    ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
    y_arr[0]  = y
    n0_arr[0] = n0_thstr
    Te_arr[0] = Te_thstr
    ni_arr[0] = ni_thstr



    i = 1
    start_time = perf_counter()
    for t in tsteps[1:]:
        t0 = perf_counter()
        # Fill model inputs 
        hyper = (j_bar, y, eta_b1, eta_b2, eta_p, n_ds, phi_dis)
        thstr = (Rth,  diverg)
        grids = (rs, ts_t[i-1], ra1_t[i-1], ra2_t[i-1], ta_t[i-1], lg, s, M_grid, rho_grid)
        disch = (ni_thstr, n0_thstr, Te_thstr)
        fclty = (n0_fac, Rch, Lch, Ns, Ms, rho_C)
        opcon = (IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi)
   
        # Calculate new accel grid radius 
        ra_ds, ra_us = accel_radius_erosion(dt, hyper, thstr, grids, disch, fclty, opcon) 
        ta_t[i]      = accel_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ts_t[i]      = screen_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ra1_t[i] = ra_ds
        ra2_t[i] = ra_us
        # Electron Backstreaming 
        rb      = min(ra1_t[i], ra2_t[i])*(beam_area_model(j_bar=j_bar)**(1/2))
        rb_t[i] = rb
        ra_avg  = np.average([ra_us, ra_ds], weights=[w,1-w])
        Vsp     = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
        dv      = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb*(1e-3), 2*ra_avg*(1e-3))
        VebsWirz_t[i] = -dv + find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )
        # Update Clausing Factor 
        CF = simpleClausingFactor(ra=min(ra1_t[i], ra2_t[i]), s=s)
        # Update discharge chamber plasma parameters
        ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
        y_arr[i]  = y
        n0_arr[i] = n0_thstr
        Te_arr[i] = Te_thstr
        ni_arr[i] = ni_thstr
        i+=1
        if print_iters == True:
            print(f"")
            print(f"Vsp = {Vsp:0.3f}, dv = {dv:0.3f}")
            print(f'ra_min = {ra_ds}, rb = {rb}')
            print(f"Clausing Factor = {CF}")
            print(f"Iteration time: {perf_counter() - t0:0.4f}s")
        else:
            pass
    
    # Present results
    print(f"Total Runtime: {perf_counter() - start_time: 0.5f} s")
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    ax = axs[0]
    ax.plot(tsteps, ra1_t, 'k-', label=r'$r_{a,ds}$')
    ax.plot(tsteps, ra2_t, 'k--', label=r'$r_{a,up}$')
    ax.plot(tsteps, rb_t, 'r-', label=r'$r_b$')
    ax.set_ylabel(r'$[mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[1]
    ax.plot(tsteps, ta_t, 'k-', label = r"$t_a$")
    ax.plot(tsteps, ts_t, 'k--', label = r"$t_s$")
    ax.set_title('Grid thickness')
    ax.set_xlabel(r'$Time [khr]$', fontsize=18)
    ax.set_ylabel(r'$[mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()

    # Plot discharge parameters
    fs = 12
    use_color = 'k'
    xlabel = r'Time [$khr$]'
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12,8))
    ax = axs[0][0]
    ax.plot(tsteps, y_arr, color=use_color)
    ax.set_title(r'Double-single ion ratio ($\gamma$)')
    #ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('', fontsize=fs)
    ax.grid(which='both')

    ax = axs[0][1]
    ax.plot(tsteps, n0_arr, color=use_color)
    ax.set_title(r'Neutral density ($n_0$)')
    #ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
    ax.grid(which='both')

    ax = axs[1][0]
    ax.plot(tsteps, Te_arr, color=use_color)
    ax.set_title(r'Electron temperature ($T_e$)')
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$eV$', fontsize=fs)
    ax.grid(which='both')
 
    ax = axs[1][1]
    ax.plot(tsteps, ni_arr, color=use_color)
    ax.set_title(r'Ion density ($n_i$)')
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
    ax.grid(which='both')

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(tsteps, abs(VebsWirz_t), 'k')
    ax.set_xlabel(r'Time $[khr]$', fontsize=fs)
    ax.set_ylabel(r'$|V_{ebs}|$', fontsize=fs)
    ax.grid(which='both')
    plt.show()

    return tsteps, ra1_t, ra2_t, ta_t, ts_t

def kwargs_simple_erosion_model(simulation_duration, step_size, print_iters=False,
                                eta_p=0.004, eta_b1=0.75, n_ds=10,      Rebs=0.001,       w=0.5,
                                lg=0.36,     rs = 1.91/2, ra_us=1.14/2, ra_ds=1.14/2,     s=2.24,           
                                ta=0.51,     ts=0.38,     Rth=0.15,     M_grid=95.95,     rho_grid=0.01022, 
                                diverg=20,   phi_p=5,     n0_fac=2e18,  Rch=2,            Lch=3,       
                                Ns=6,        Ms=12.011,   rho_C=2.25e6, Te_beam=1,        nbp=5e16,    
                                IB=1.76,     Ib=2.7e-4,   Vd=1100,      Va=-180,          phi_dis=26,  
                                Ni=54,       Mi=131.27,   Ec=400,       E_cex=400):

    eta_b2   = 1-eta_b1
    ta0      = ta
    ts0      = ts
    


    # initialize results arrays
    dt      = step_size
    T_end   = simulation_duration
    tsteps  = np.arange(0, T_end+dt, dt)
    ra1_t   = np.zeros(len(tsteps))
    ra2_t   = np.zeros(len(tsteps))
    rb_t    = np.zeros(len(tsteps))
    ta_t    = np.zeros(len(tsteps))
    ts_t    = np.zeros(len(tsteps))

    VebsWirz_t = np.zeros(len(tsteps))
    y_arr  = np.zeros(len(tsteps))
    n0_arr = np.zeros(len(tsteps))
    Te_arr = np.zeros(len(tsteps))
    ni_arr = np.zeros(len(tsteps))
    
    
    CF = simpleClausingFactor(ra=min(ra_ds, ra_us), s=s)
    # Update discharge chamber plasma parameters at t=0
    ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
    y_arr[0]  = y
    n0_arr[0] = n0_thstr
    Te_arr[0] = Te_thstr
    ni_arr[0] = ni_thstr

    # Calculates Secondary variables
    leff  = wgw.eff_grid_gap(lg*(1e-3), ts0*(1e-3), 2*(rs*(1e-3)))
    As    = 2*pi*(rs*1e-3)**2 
    Ibohm = As*bohm_current_density(ni_thstr, Te_thstr, Mi)
    jCL   = j_CL(Mi, Vd-Va, leff)
    ICL   = As*jCL
    j_bar = Ib /( As * j_CL(Mi, Vd-Va, leff))

    # Populates results at t=0
    ra1_t[0] = ra_ds
    ra2_t[0] = ra_us
    rb_t[0]  = np.average([ra_us, ra_ds], weights=[w, 1-w])*(beam_area_model(j_bar=j_bar)**(1/2))
    ta_t[0]  = ta0
    ts_t[0]  = ts0

    
    # Electron Backstreaming model
    w1, w2        = potential_ratio(ra2_t[0], ra1_t[0])
    ra_avg        = np.average([ra1_t[0], ra2_t[0]], weights=[w1,w2])
    Vsp           = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
    # Verify that Vd+abs(Va) is the appropriate potential in his calculation.
    dv            = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb_t[0]*(1e-3), 2*ra_avg*(1e-3))
    VebsWirz_t[0] = -dv + find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs)

    if print_iters == True:
        print(f"j_bar = {j_bar:0.4f}")
        print(f"I_cl  = {ICL :0.4e}")
        print(f"Ib    = {Ib:0.2e}")
        print(f"Ibohm = {Ibohm:0.2e}")
        print(f"Ib/IBohm = {Ib/Ibohm:0.2f}")
        print(f"Vsp = {Vsp:0.3f}, dv = {dv:0.3f}, Vebs = {VebsWirz_t[0]:0.3f}")
        print(f'ra_min = {ra1_t[0]:0.3f}, ra_max = {ra2_t[0]:0.3f}, rb = {rb_t[0]:0.3f}')
        print(f"Clausing Factor = {CF:0.3f}")

    i = 1
    T = 0
    start_time = perf_counter()
    for t in tsteps[1:]:
        T+=dt
        t0 = perf_counter()
        # Fill model inputs 
        hyper = (j_bar, y, eta_b1, eta_b2, eta_p, n_ds, phi_dis)
        thstr = (Rth,  diverg)
        grids = (rs, ts_t[i-1], ra1_t[i-1], ra2_t[i-1], ta_t[i-1], lg, s, M_grid, rho_grid)
        disch = (ni_thstr, n0_thstr, Te_thstr)
        fclty = (n0_fac, Rch, Lch, Ns, Ms, rho_C)
        opcon = (IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi)
   
        # Calculate new accel grid radius 
        ra1_t[i], ra2_t[i] = accel_radius_erosion(dt, hyper, thstr, grids, disch, fclty, opcon) # returns ra_ds, ra_us
        ta_t[i]            = accel_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ts_t[i]            = screen_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
    
        
        # Electron Backstreaming 
        #rb      = min(ra1_t[i], ra2_t[i])*(beam_area_model(j_bar=j_bar)**(1/2))
        rb      = np.average([ra1_t[i], ra2_t[i]], weights=[w, 1-w])*(beam_area_model(j_bar=j_bar)**(1/2))
        rb_t[i] = rb
        w1, w2 = potential_ratio(ra1_t[i], ra2_t[i])
        ra_avg  = np.average([ra1_t[i], ra2_t[i]], weights=[w1, w2])
        Vsp     = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, Mi, Vd)
        dv      = wgw.spacecharge_effect(Ib, Vd, Vsp, Mi, 2*rb_t[i]*(1e-3), 2*ra_avg*(1e-3))
        #VebsWirz_t[i] =  find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, j_bar, Ib*Rebs )
        VebsWirz_t[i] = -dv + find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs)
        
        # Update Clausing Factor 
        CF = simpleClausingFactor(ra=min(ra1_t[i], ra2_t[i]), s=s)
        # Update discharge chamber plasma parameters
        ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
        y_arr[i]  = y
        n0_arr[i] = n0_thstr
        Te_arr[i] = Te_thstr
        ni_arr[i] = ni_thstr
        
        #if print_iters == True:
        #    print(f"")
        #    print(f"i={i}, T = {T:0.2f} khr")
        #    print(f"Vsp = {Vsp:0.3f}, dv = {dv:0.3f}, Vebs = {VebsWirz_t[i]:0.3f}")
        #    print(f'ra_min = {ra_ds:0.3f}, ra_max = {ra_us:0.3f}, rb = {rb:0.3f}')
        #    print(f"Clausing Factor = {CF:0.3f}")
        #    print(f"Iteration time: {perf_counter() - t0:0.4f}s")

        i+=1
    # Present results
    print(f"Run time: {perf_counter()-start_time:0.6f}s")
    print("")
    return tsteps, ra1_t, ra2_t, rb_t, ta_t, ts_t, VebsWirz_t, y_arr, ni_arr, n0_arr, Te_arr
   
def plot_results(t, ra1, ra2, rb, ta, ts, y, n0, ni, Te, Vebs ):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
        ax = axs[0]
        ax.plot(t, ra1, 'k-', label=r'$r_{a,ds}$')
        ax.plot(t, ra2, 'k--', label=r'$r_{a,up}$')
        ax.plot(t, rb, 'r-', label=r'$r_b$')
        ax.set_ylabel(r'$[mm]$', fontsize=18)
        ax.grid(which='both')
        ax.legend()
        ax = axs[1]
        ax.plot(t, ta, 'k-', label = r"$t_a$")
        ax.plot(t, ts, 'k--', label = r"$t_s$")
        ax.set_title('Grid thickness')
        ax.set_xlabel(r'$Time [khr]$', fontsize=18)
        ax.set_ylabel(r'$[mm]$', fontsize=18)
        ax.grid(which='both')
        ax.legend()

        # Plot discharge parameters
        fs = 12
        use_color = 'k'
        xlabel = r'Time [$khr$]'
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12,8))
        ax = axs[0][0]
        ax.plot(t, y, color=use_color)
        ax.set_title(r'Double-single ion ratio ($\gamma$)')
        #ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel('', fontsize=fs)
        ax.grid(which='both')

        ax = axs[0][1]
        ax.plot(t, n0, color=use_color)
        ax.set_title(r'Neutral density ($n_0$)')
        #ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
        ax.grid(which='both')

        ax = axs[1][0]
        ax.plot(t, Te, color=use_color)
        ax.set_title(r'Electron temperature ($T_e$)')
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel(r'$eV$', fontsize=fs)
        ax.grid(which='both')
    
        ax = axs[1][1]
        ax.plot(t, ni, color=use_color)
        ax.set_title(r'Ion density ($n_i$)')
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel(r'$m^{-3}$', fontsize=fs)
        ax.grid(which='both')

        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.plot(t, abs(Vebs), 'k')
        ax.set_xlabel(r'Time $[khr]$', fontsize=fs)
        ax.set_ylabel(r'$|V_{ebs}|$', fontsize=fs)
        ax.grid(which='both')
        #plt.show()

def nstar_elt():
    # Throttle conditions are a consolidation of Wirz 2011; Fig. 1 & Table 1 and Sengupta 2003; Table 1
    throttle_conditions = {"TH15" : {"IB": 1.76, "Ib": 0.27e-3, "Vb": 1100, "Va":-180, "n0_local": 3.50e17},
                           "TH12" : {"IB": 1.49, "Ib": 0.25e-3, "Vb": 1100, "Va":-180, "n0_local": 3.15e17},
                           "TH8"  : {"IB": 1.10, "Ib": 0.20e-3, "Vb": 1100, "Va":-180, "n0_local": 2.80e17},
                           "TH5"  : {"IB": 0.81, "Ib": 0.15e-3, "Vb": 1100, "Va":-150, "n0_local": 2.45e17},
                           "TH0"  : {"IB": 0.51, "Ib": 0.10e-3, "Vb":  650, "Va":-180, "n0_local": 2.10e17}}
    ## Simulations Conditions 
    dt = 0.1
    # Facility - Sengupta 2003; Pg 3
    Rch = 3
    Lch = 10

    test_plan = ["TH12", "TH15", "TH8", "TH15", "TH0", "TH15", "TH5"]
    end_times = [0.5, 4.7, 10.5, 15.6, 21.3, 25.7, 30.4] # kHr
    start_times = [0] + end_times[:-1]
    
    tsteps    = np.array([])
    ra_ds_elt = np.array([])
    ra_us_elt = np.array([]) 
    rb_elt    = np.array([])
    ta_elt    = np.array([]) 
    ts_elt    = np.array([]) 
    Vebs_elt  = np.array([])

    for seg, th in enumerate(test_plan):
        segment_duration = end_times[seg] - start_times[seg]
        print(f"{th}")
        if seg == 0:
            t, ra_ds, ra_us, rb, ta, ts, Vebs, _, _, _, _ = kwargs_simple_erosion_model(segment_duration, dt, print_iters=True,
                                                                                        IB     = throttle_conditions[th]['IB'], 
                                                                                        Ib     = throttle_conditions[th]['Ib'],
                                                                                        Vd     = throttle_conditions[th]['Vb'],
                                                                                        Va     = throttle_conditions[th]['Va'],
                                                                                        nbp    = throttle_conditions[th]['n0_local'], 
                                                                                        eta_p = 0.004, eta_b1=0.75, n_ds=10, w=0.5)#,
                                                                                        #Rch=Rch, Lch=Lch)   
            tsteps = np.append(tsteps, t)

        else:
            t, ra_ds, ra_us, rb, ta, ts, Vebs, _, _, _, _ = kwargs_simple_erosion_model(segment_duration, dt, print_iters=True,
                                                                                        IB     = throttle_conditions[th]['IB'], 
                                                                                        Ib     = throttle_conditions[th]['Ib'],
                                                                                        Vd     = throttle_conditions[th]['Vb'],
                                                                                        Va     = throttle_conditions[th]['Va'],
                                                                                        nbp = throttle_conditions[th]['n0_local'],
                                                                                        ra_ds = ra_ds[-1], ra_us=ra_us[-1], ta=ta[-1], ts=ts[-1], 
                                                                                        eta_p = 0.004, eta_b1=0.75, n_ds=10, w=0.5)#,
                                                                                        #Rch=Rch, Lch=Lch) 
            tsteps = np.append(tsteps, tsteps[-1]+t)
        ra_ds_elt = np.append(ra_ds_elt, ra_ds)
        ra_us_elt = np.append(ra_us_elt, ra_us)
        rb_elt    = np.append(rb_elt, rb)
        ta_elt    = np.append(ta_elt, ta)
        ts_elt    = np.append(ts_elt, ts)
        Vebs_elt  = np.append(Vebs_elt, Vebs)
    return tsteps, ra_ds_elt, ra_us_elt, rb_elt, ta_elt, ts_elt, Vebs_elt, None, None, None, None

def sim_and_show_nstar_elt():
    elt_wirz_vebs = [[140, 141], [150, 166], [150, 155], [187, 200],[0,0], [200, 240], [190, 195]] 
    t, ra_ds, ra_us, rb, ta, ts, Vebs, _, _, _, _ = nstar_elt()
    print(f"Post test min(ra): {min(ra_ds[-1], ra_us[-1]):0.3f} mm")
    print(f"Post test Vebs: {Vebs[-1]:0.3f} V")
    
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Extended Life Test -- ROM Results")
    fig.tight_layout()
    ax_r = plt.subplot(2,2,1)
    ax_t = plt.subplot(2,2,3)
    ax_V = plt.subplot(1,2,2)
    axes = [ax_r, ax_t, ax_V]

    end_times = [0.5, 4.7, 10.5, 15.6, 21.3, 25.7, 30.4] # kHr
    start_times = [0] + end_times[:-1]
    time, ra_elt = grab_data_general("erosion_rom/wirz-time_dependent_fig11.csv", ["T", 'ra'])
    for i in range(len(start_times)):
        axes[2].plot([start_times[i], end_times[i]], elt_wirz_vebs[i], 'r-')

    axes[2].plot(t, abs(Vebs), 'k-')
    
    axes[2].set_ylim([110, 250])
    axes[2].set_xlabel("Time [khr]")
    axes[2].set_ylabel(r"$|V_{ebs}$|", color='blue')
    axes[0].set_ylabel(r'$d_b [mm]$')#, color='red')
    
    axes[2].grid(which='both')

    axes[0].plot(t, 2*ra_ds, 'k-', label=r'$r_{a,ds}$')
    axes[0].plot(t, 2*ra_us, 'k-.', label=r'$r_{a,us}$')
    axes[0].plot(time, ra_elt, 'b-*', label=r'ELT $min(r_a)$')
    #axes[0].plot(t, 2*rb, 'r-', label=r'$r_b$')
    axes[0].legend()
    axes[0].grid(which='both')
    #axes[0].set_xlabel(r"Time $[khr]$")
    axes[0].set_ylabel(r"Diameter $[mm]$")

    axes[1].plot(t, ta, 'k-', label=r'$t_{a}$')
    axes[1].plot(t, ts, 'k-.', label=r'$t_{s}$')
    axes[1].legend()
    axes[1].grid(which='both')
    axes[1].set_xlabel(r"Time $[khr]$")
    axes[1].set_ylabel(r"Thickness $[mm]$")
    plt.show()

def main():
    #t, ra_ds, ra_us, ta, ts = json_simple_erosion_model('sample.json', 8.2, 0.1)
    T_ldt, Vebs_ldt = grab_data_general("erosion_rom/ldt_vebs/ldt_data_fig23.csv", ['T', 'Vebs'])
    t, ra_ds, ra_us, rb, ta, ts, Vebs, _, _, _, _ = kwargs_simple_erosion_model(8.2, 0.1, w=1, Rebs=0.001)
    # Plot Results
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, abs(Vebs), 'r-', label='ROM Results')
    #ax.plot(T_ldt/1000, abs(Vebs_ldt), "k*", label='LDT Data (Fig. 23)')
    ax.legend()
    ax.set_xlabel("Time [khr]")
    ax.set_ylabel(r"$|V_{ebs}$|")
    ax.set_ylim([110, 250])
    ax.grid(which='both')
    print(f"Post test min(ra): {min(ra_ds[-1], ra_us[-1]):0.3f} mm")
    print(f"Post test Vebs: {Vebs[-1]:0.3f} V")

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, ra_ds, 'k-', label=r'$r_{a,ds}$')
    ax[0].plot(t, ra_us, 'k-.', label=r'$r_{a,us}$')
    ax[0].plot(t, rb, 'r-', label=r'beam radius')
    ax[0].legend()
    ax[0].grid(which='both')
    ax[0].set_xlabel(r"Time $[khr]$")
    ax[0].set_ylabel(r"Accel Radius $[mm]$")

    ax[1].plot(t, 100*ta/ta[0], 'k-', label=r'$t_{a}$')
    ax[1].plot(t, 100*ts/ts[0], 'k-.', label=r'$t_{s}$')
    ax[1].legend()
    ax[1].grid(which='both')
    ax[1].set_xlabel(r"Time $[khr]$")
    ax[1].set_ylabel(r"% Nominal thickness")
    

    plt.show()

    #np.savetxt('erosion_rom/erosion_visualize/grid_geometry.csv', np.array([t, ra_ds, ra_us, ta, ts]).T, delimiter=',')

    return

if __name__ == "__main__":
    sim_and_show_nstar_elt()
  
