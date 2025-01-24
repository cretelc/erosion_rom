## Dependencies
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
'''
def F(Va, **kwargs): #, ra, nbp, cbar, Te, phi_sp, phi_bp):
    k1 = 2 * pi * (kwargs['ra']**2) * q * kwargs['Te']
    k2 = kwargs['nbp']*kwargs['cbar'] / 4
    k3 = np.exp((kwargs['phi_sp'] - kwargs['phi_bp'])/kwargs['Te'])
    C = k1*k2*k3
    #return C * (np.exp((Va-kwargs['phi_sp'])/kwargs['Te']) - 1) / (Va - kwargs['phi_sp']) - kwargs['Iebs_limit']
    return -C  / (Va - kwargs['phi_sp']) - kwargs['Iebs_limit']

def dF(Va, **kwargs): #ra, nbp, cbar, Te, phi_sp, phi_bp):
    k1 = 2 * pi * ((kwargs['ra'])**2) * q * kwargs['Te']
    k2 = kwargs['nbp']*kwargs['cbar'] / 4
    k3 = np.exp((kwargs['phi_sp'] - kwargs['phi_bp'])/kwargs['Te'])
    C = k1*k2*k3
    
    g  = np.exp((Va - kwargs['phi_sp'])/kwargs['Te']) - 1
    g = -1
    h  = 1/(Va - kwargs['phi_sp'])
    Dg = (1/kwargs['Te']) * np.exp((Va - kwargs['phi_sp']) / kwargs['Te'])
    Dh = -1 / ((Va - kwargs['phi_sp'])**2)

    return C * (h*Dg + Dh*g)

def newton(x0, f, Df, error, maxiter, **kwargs):
    xn = x0
    for i in range(0, maxiter):
        fxn  = f(xn, **kwargs)
        Dfxn = Df(xn, **kwargs)
        if abs(fxn) < error:
            print(f"Solution, {xn}, found after {i} iterations.")
            return xn
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn / Dfxn
    print(f'Hit max iterations without finding solution: {xn}')
    return None

def find_Vebs(Va, ra, Te, Vsp, Vbp, nbp, cbar, Ilimit):
    for_kwargs = {'ra': ra, 'Te':Te, 'phi_sp': Vsp, 'phi_bp': Vbp, 'nbp': nbp, 'cbar': cbar, 'Iebs_limit': Ilimit}
    Vebs = newton(Va, F, dF, 1e-7, 100000, **for_kwargs)
    return Vebs
'''
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
    j_bar, Ibar, y, P1, P2, eta_cex, n_ds, phi_p    = hyperparams
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
    M_kg    = Mi / (1000 * N_A)
    lg_kg   = lg * (1e-3)
    n0_mm3  = n0_dis * (1e-9)

    erosion_control = np.array([P1, P2])
    ra              = np.array([ra1, ra2])

    jCL              = j_CL(M_kg, Vd-Va, lg_kg)                # -> A m-2 this is overpredicting Ib but it's ok
    jBohm            = bohm_current_density(n0_dis, Te_dis, M_kg)
    #jb               = j_bar * jBohm                            #j_bar * jBohm
    jb               = j_bar * jCL                            #j_bar * jBohm
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)            # -> m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)           # -> m2
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)          # -> atoms/ion; produces reasonable results
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)        # -> atoms/ion; produces reasonable results
    rb               = min(ra) * beam_area_model(j_bar)**(1/2)  # -> mm
    Vbeam            = beam_volume_trap(lg, rs, rb)             # -> mm3
    Gamma_CEX        = cex_generation_rate(jb, n0_mm3, sigma_p, Vbeam)     # -> ions/s, Gamma^C_CEX 

    sputter_flux     = erosion_control * cex_erosion_flux(Gamma_CEX, y, Y_p, Y_pp, sigma_p, sigma_pp) # atoms/s
    dra_dt_0         = sputter_flux * M_grid / (N_A * rho_grid) # mm/s
    # Convert to mm/khr
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
    j_bar, Ibar, y, _, _, eta_cex, n_ds, phi_p    = hyperparams
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
    j_bar, Ibar, y, _, _, eta_cex, n_ds, phi_p    = hyperparams
    rs, ts, _, _, ta, lg, s, M_grid, rho_grid         = grids
    # Pull arguments
    Rth, div                                = thruster
    ni_dis, n0_dis, Te_dis                  = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C         = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi       = operating_conditions

    # convert special variables to their preferred units
    M_kg       = Mi / ( 1000 * N_A)
    lg_kg      = lg * (1e-3)
    n0_mm3     = n0_dis * (1e-9)
    rho_grid_m = rho_grid * (1e9) # g/mm3 -> g/m3

    jBohm        = bohm_current_density(ni_dis, Te_dis, M_kg)           # A m-2
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



## Functions
def erosion_rate(flux, sputter_yield, target_atomic_weight, target_density):
    return flux * sputter_yield * (target_atomic_weight / (target_density * N_A))

def cex_erosion_flux(flux_cex, single_double_ratio, single_yield, double_yield, single_xsection, double_xsection, carbon_surface_coverage=0):
    A = flux_cex * (1 - carbon_surface_coverage) / (1+single_double_ratio)
    B = (single_double_ratio/2) * (double_xsection/single_xsection)
    return A * (single_yield + B*double_yield)

# Current Densities
def bohm_current_density(n0, Te_eV, M):
    Te = Te_eV * q
    return 0.606 * n0 * q * (Te/M)**0.5

def j_CL(M , V, d):
    ''' Calculates the Child-Langmuir current density
    
    Arguments
    M -- ion mass, kg 
    V -- total potential, Vd - Va
    d -- intergrid distance
    '''
    return (4 * eps0 / 9) * (2 * q / M)**(0.5) * (V**(3/2) / (d**2))


def double_single_sputter_yield(j, double_single_ion_ratio, single_yield, double_yield):
    ''' Calculates sputter yield from single and doubley charged ions, generally'''
    return ( j / ( q*(1+double_single_ion_ratio) )) * (single_yield + (double_single_ion_ratio/2)*double_yield)


def simple_erosion_model():
    # Physical constants and general material properties 
    ## User set values
    # hyperparameters ---------------------------------------
    j_bar    = 0.5     # normalized beamlet current
    I_bar    = 1.8      # beamlet-bohm current ratio
    eta_p    = 0.004    # cex ion impingement probability in the plume
    eta_b1   = 0.75     # upstream cex ion impingement probability between the grids
    eta_b2   = 1-eta_b1 # downstream cex ion impingement probability between grids
    n_ds     = 10       # thruster radius's downstream 
    Rebs     = 0.001    # electron backstreaming ratio limit
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
    # facility ----------------------------------------------
    n0_fac   = 2e18     # facility number density, m-3
    Rch      = 2        # chamber radius, m
    Lch      = 3        # chamber length, m
    Ns       = 6        # chamber wall material atomic number
    Ms       = 12.011   # chamber wwall material atomic mass, g/mol
    rho_C    = 2.25e6   # redeposited material density, g/m3
    Te_beam  = 2        # beam electron temperature, eV
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
    
    # ELT data
    #xdata, ydata = grab_data()

    ## Holding for testing variables 
    nbp = 5e16


    # initialize results arrays
    dt     = 1
    T_end  = 20
    tsteps = np.arange(0, T_end+dt, dt)
    ra1_t   = np.zeros(len(tsteps))
    ra2_t   = np.zeros(len(tsteps))
    ta_t   = np.zeros(len(tsteps))
    ts_t   = np.zeros(len(tsteps))

    VebsWirz_t = np.zeros(len(tsteps))
    VebsWGW_t = np.zeros(len(tsteps))
    y_arr  = np.zeros(len(tsteps))
    n0_arr = np.zeros(len(tsteps))
    Te_arr = np.zeros(len(tsteps))
    ni_arr = np.zeros(len(tsteps))
    # Population results at t=0
    ra1_t[0] = ra0
    ra2_t[0] = ra0
    ta_t[0]  = ta0
    ts_t[0]  = ts0
    rb0      = 0.9*ra0
    rb = ra0*(beam_area_model(j_bar=j_bar)**(1/2))
    
    VebsWGW_t[0] = wgw.electron_backstreaming(phi_p, Vd, Va, Ib, Te_beam, rs, ra0, ts0, ta0, lg, rb, 131, Rebs )
    Vsp = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, 131, Vd)
    dv = wgw.spacecharge_effect(Ib, Vd, Vsp, 131, 2*rb*(1e-3), 2*ra0*(1e-3))
    VebsWirz_t[0] =  -dv+find_Vebs(Va, ra0*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )
    CF = simpleClausingFactor(ra=ra0, s=s)
    # Update discharge chamber plasma parameters
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
        hyper = (j_bar, I_bar, y, eta_b1, eta_b2, eta_p, n_ds, phi_dis)
        thstr = (Rth,  diverg)
        grids = (rs, ts_t[i-1], ra1_t[i-1], ra2_t[i-1], ta_t[i-1], lg, s, M_grid, rho_grid)
        disch = (ni_thstr, n0_thstr, Te_thstr)
        fclty = (n0_fac, Rch, Lch, Ns, Ms, rho_C)
        opcon = (IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi)
   
        # Calculate new accel grid radius 
        ra_ds, ra_us = accel_radius_erosion(dt, hyper, thstr, grids, disch, fclty, opcon) 
        ta_t[i]            = accel_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ts_t[i]            = screen_thickness_erosion(dt, hyper, thstr, grids, disch, fclty, opcon)
        ra1_t[i] = ra_ds
        ra2_t[i] = ra_us
        print(f"ra1 = {ra1_t[i]:0.3f}, ra2 = {ra2_t[i]:0.3f}")
        # Electron Backstreaming 
        rb = min(ra1_t[i], ra2_t[i])*(beam_area_model(j_bar=j_bar)**(1/2))
        #rb = 0.9*min(ra1_t[i], ra2_t[i])
        w = 0.5
        ra_avg = np.average([ra_us, ra_ds], weights=[w,1-w])
        Vsp = wgw.retarding_Vsp(phi_p, Te_beam, Rebs, 131, Vd)
        dv = wgw.spacecharge_effect(Ib, Vd, Vsp, 131, 2*rb*(1e-3), 2*ra_avg*(1e-3))
        print(f"dV = {dv:0.2f} V")
        print(f"Vsp = {Vsp:0.2f} V")
        VebsWirz_t[i] = -dv + find_Vebs(Va, ra_avg*(1e-3), Te_beam, Vsp, phi_p, nbp, Ib*Rebs )
        VebsWGW_t[i] =  wgw.electron_backstreaming(phi_p, Vd, Va, Ib, Te_beam, 
                                                  rs*(1e-3), ra_avg*(1e-3), ts_t[i]*(1e-3), ta_t[i]*(1e-3),
                                                  lg*(1e-3), rb*(1e-3), 131, Rebs )
        # Calculate new Clausing Factor with updated accel grid radius
        CF = simpleClausingFactor(ra=min(ra1_t[i], ra2_t[i]), s=s)
        # Update discharge chamber plasma parameters
        ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
        y_arr[i]  = y
        n0_arr[i] = n0_thstr
        Te_arr[i] = Te_thstr
        ni_arr[i] = ni_thstr
        
        i+=1
        print(f"Iteration time: {perf_counter() - t0:0.4f}s")
    # Show results
    print(f"Total Runtime: {perf_counter() - start_time: 0.5f} s")
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    ax = axs[0]
    ax.plot(tsteps, ra1_t, 'k-', label=r'$r_{a,ds}$')
    ax.plot(tsteps, ra2_t, 'k--', label=r'$r_{a,up}$')
    #ax.plot(xdata, ydata, 'k*', label = 'ELT Data')
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
    ax.plot(tsteps, abs(VebsWirz_t), 'r', label='Wirz Integration Method')
    ax.plot(tsteps, abs(VebsWGW_t), 'g', label='WGW Analytical Method')
    ax.set_xlabel(r'Time $[khr]$', fontsize=fs)
    ax.set_ylabel(r'$|V_{ebs}|$', fontsize=fs)
    ax.grid(which='both')
    ax.legend()
    plt.show()

    return tsteps, ra1_t, ra2_t, ta_t, ts_t



        
def main():
    t, ra_ds, ra_us, ta, ts = simple_erosion_model()
    #np.savetxt('erosion_rom/erosion_visualize/grid_geometry.csv', np.array([t, ra_ds, ra_us, ta, ts]).T, delimiter=',')
    #Va = -np.linspace(10, 5e3, 1000)
    #Vebs_search(Va, 1.14/2, 2, -12, 15, 1e14, 950000, (1e-3)*(0.27e-3))
    return





if __name__ == "__main__":
    main()
    #iter_erosion_model()
    #iter_thickness_model()
