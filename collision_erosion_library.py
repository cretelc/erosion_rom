import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.constants import Avogadro as N_A
from scipy.constants import elementary_charge as q
from scipy.constants import Boltzmann as kB
from scipy.constants import epsilon_0 as eps0
from scipy.optimize import curve_fit
import eckstein_yield as eck
from uncertainties import ufloat
from uncertainties.umath import log as ulog
from uncertainties.umath import log10 as ulog10
import eckstein_yield as ey
from random import uniform as uni
from clausing import simpleClausingFactor
import pandas as pd


## Carbon Backsputtering Functions
def plume_volume(Rth, n_ds = 10, div = 20):
    div_rad = div * np.pi / 180
    #print(div_rad)
    return ( (np.pi) * Rth**3 ) * ( ( n_ds**3 / 3) * np.tan( div_rad )**2 + ( n_ds**2 ) * np.tan( div_rad ) + n_ds )

def redep_flux_Xe_C(Ib, Rth, E):
    Y   = ey.calculate_yield(E, 6, 54, 12.011, 131.29, [4, 0.8, 1.8, 21])
    Ath = np.pi * Rth**2
    carbon_flux = Y*Ib / (Ath * q)
    return carbon_flux

def sputterant_thickness_rate(IB, Rch, Ns, Ni, Ms, Mi, rho_s, E, params = [4, 0.8, 1.8, 21], sc=1):
    Y = 0.5
    Y = ey.calculate_yield(E, Ns, Ni, Ms, Mi, params)
    Ach = np.pi * Rch*Rch
    return sc*(Y*IB/(q * Ach)) * Ms / (N_A * rho_s)

def sputterant_redep_rate(IB, Rch, Rth, Ns, Ni, Ms, Mi, rho_s, E, params = [4, 0.8, 1.8, 21], sc=1):
    Y = ey.calculate_yield(E, Ns, Ni, Ms, Mi, params)
    Ath = np.pi * Rth * Rth
    Ach = np.pi * Rch * Rch
    s_avg = (Y * IB / q) * (Ath / Ach)
    return s_avg

def cex_generation_rate(IB, Rth, n0_fac, sigma_p):
    Vbeam = plume_volume(Rth)
    return IB * n0_fac * sigma_p * Vbeam

def carbon_surface_coverage(GammaC, ja, Y_C_Mo=0.99):
    thetac = q*GammaC / (ja * Y_C_Mo)
    return thetac

def carbon_surface_coverageV2(sdot, Icex, Y=0.99):
    Y_C_Mo = Y
    thetaC = sdot / (Icex * Y_C_Mo)
    return thetaC

def accel_thickness_erosion_model(dt, hyperparams, grid_geometry, grid_mats, model_inputs):
    # In work
    j_bar, y                    = hyperparams
    Rth, rs, ts, ra, ta, lg, s  = grid_geometry
    M_grid, rho_grid            = grid_mats
    n0, Ec, E_cex, Vd, Va, M    = model_inputs
    
    # Same as accel_radius_erosion_model()
    jCL              = j_CL(M, Vd-Va, lg*(1e-3))        # A m-2
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)    # m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)   # m2

    # updated for accel grid thickness

    Vbeam            = plume_volume(Rth, n_ds=10, div = 20)  # m2
    IB               = jCL * j_bar * (pi * Rth**2)           # A
    carbon_red_flux  = redep_flux_Xe_C(1.76, Rth, np.abs(Vd))      # = 7.57 atoms / s m2
    
    #ja = 6.78e-3 / (pi * Rth**2)
    Gamma_CEX        = cex_flux(1.76 / (pi*Rth**2), n0, sigma_p, Vbeam) # Gamma^C_CEX 
    theta_C          = carbon_surface_coverage(carbon_red_flux, ja=q*Gamma_CEX) #  using 6.78e-3A from Souas 2013 as a placeholder
    
    #Gamma_CEX        = cex_flux(1.76 / (pi*Rth**2), n0, sigma_p, Vbeam) # Gamma^C_CEX 

    # Similar to accel_radius_erosion_model() with the addition of theta_C in sputter_flux
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)
    sputter_flux     = cex_erosion_flux(Gamma_CEX, y, Y_p, Y_pp, sigma_p, sigma_pp)#, theta_C)
    dta_dt_0         = sputter_flux * M_grid / (N_A * rho_grid)
    dta_dt           = dta_dt_0*(60)*(60)*(1000) # m / khr
    t_erode          = ta - dt* (dta_dt)
    return t_erode

def accel_thickness_erosion_modelV2(dt,  hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
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
    j_bar, Ibar, y, eta_cex, n_ds           = hyperparams
    Rth, div                                = thruster
    rs, ts, ra, ta, lg, s, M_grid, rho_grid = grids
    ni_dis, n0_dis, Te_dis                  = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C         = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi       = operating_conditions
    
    # Same as accel_radius_erosion_model()
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)    # m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)   # m2

    # updated for accel grid thickness ---
    
    Vbeam            = plume_volume(Rth, n_ds=10, div = 20)  # m2  
    s_avg            = sputterant_redep_rate(IB, Rch, Rth, Ns, 54, Ms, 131.27, rho_C, Vd)
    I_CEX            = cex_generation_rate(IB, Rth, n0_fac, sigma_p)
    theta_C          = carbon_surface_coverageV2(s_avg, eta_cex*I_CEX/q) 
    Y_p              = ey.calculate_yield_Xe_Mo(abs(Va))          # -> atoms/ion; produces reasonable results
    Y_pp             = ey.calculate_yield_Xe_Mo(2*abs(Va))        # -> atoms/ion; produces reasonable results
    sputter_flux     = cex_erosion_flux(eta_cex*I_CEX/q, y, Y_p, Y_pp, sigma_p, sigma_pp)#, carbon_surface_coverage=theta_C) # atoms/s
    dta_dt           = sputter_flux * M_grid / (N_A * rho_C)
    dta_dt           = dta_dt*1000*(60)*(60)*(1000) # mm / khr
    t_erode          = ta - dt * dta_dt 
    #print(f"carbon_red_flux = {carbon_red_flux:0.3e} [m-2 s-1]") # target is 2.19e12 cm-2 s-1
    #print(f"carbon_red_flux = {carbon_red_flux * (1/100)**2:0.3e} [cm-2 s-1]") # target is 2.19e12 cm-2 s-1
    return t_erode
    
def grab_data():
    measured_da_data = pd.read_csv('collision_erosion_model/wirz-time_dependent_fig11.csv', names=['T', 'da'])
    xdata      = measured_da_data['T'].to_numpy()
    ydata      = measured_da_data['da'].to_numpy() / 2 # da -> ra
    return xdata, ydata

## Functions
def beam_volume_trap(h, r1, r2):
    ''' Calculates the beam volume between the screen and accel grid using 
        approximating the beam as a truncated trapezoid.

        returns 
        beam volume, assuming units of all parameters is the same    
    '''
    return ( pi * h / 3 ) * (r2**2 + r1*r2 + r1**2)

def erosion_rate(flux, sputter_yield, target_atomic_weight, target_density):
    return flux * sputter_yield * (target_atomic_weight / (target_density * N_A))

def cex_erosion_flux(flux_cex, single_doube_ratio, single_yield, double_yield, single_xsection, double_xsection, carbon_surface_coverage=0):
    A = flux_cex * (1 - carbon_surface_coverage) / (1+single_doube_ratio)
    B = (single_doube_ratio/2) * (double_xsection/single_xsection)
    return A * (single_yield + B*double_yield)

def cex_flux(j, n0, single_xsection, volume):
    return (j/q) * n0 * single_xsection * volume

def beam_area_model(j_bar=1, a=0.619, b=0.011, c=-0.045):
    return a*j_bar + (b/j_bar) + c

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

def bohm_velocity(Te, m):
    return np.sqrt(kB*Te / m)

def bohm_current_density(n0, Te_eV, M):
    Te = Te_eV * q
    return 0.606 * n0 * q * (Te/M)**0.5

def discharge_density(Ib, A, Te, m):
    ''' Calculates the discharge density assuming beam current, Ib ~2IBohm 
    currently not used
    '''
    return ( Ib / (q*A)) * np.sqrt(m / (kB*Te))

def j_CL(M , V, d):
    ''' Calculates the Child-Langmuir current density
    
    Arguments
    M -- ion mass, kg 
    V -- total potential, Vd - Va
    d -- intergrid distance
    '''
    return (4 * eps0 / 9) * (2 * q / M)**(0.5) * (V**(3/2) / (d**2))

def f(x, a, b):
    return a*np.log(1+b*x)

def dfdt(x, a, b):
    return a*b/(1 + b*x)

def calculate_parameters_from_clausing(C): 
    ni = (-1.65*C + 2.228) * (1e17)
    no = (-5.281*C + 3.050) * (1e18)
    y  = (0.2562*C + 0.0203)
    Te = (1.6636*C + 6.3558)
    return ni, no, y, Te

def accel_grid_erosion_func(t, a, b):

    dPdt = dfdt(t, a, b)#a*b / (1 + b*t)

    j_bar = 1 #ufloat(1, 0.2)
    n0_sa = 1e19
    double_ion_ratio = 0.08 #ufloat(0.08, 0.05)
    lg     = 0.7
    rs     = (1.91/2)# * (1e-3)
    ra     = (1.14/2)# * (1e-3)
    Vd     = 1500
    Va     = -190
    M      = 2.18e-25 # kg, xenon particle mass
    Ec     = 400, #ufloat(400, 200) # using the average of the energy
    E_cex  = 500
    M_grid = 95.95 # g/mol
    rho_grid = 10.22 * (100)**3 # g/cm3
    
    jCL              = j_CL(M, Vd-Va, lg*(1e-3))
    sigma_p          = xe_xsections(Ec, "+")*(1e-20) # m^2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20) # m^2
    rb               = ra * beam_area_model(j_bar)**(1/2)
    nominal_beam_vol = beam_volume_trap(lg, rs, rb)
    Vbeam            = nominal_beam_vol
    Gamma_CEX        = dPdt*cex_flux(jCL*j_bar, n0_sa, sigma_p, Vbeam) # Gamma^C_CEX 
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)
    sputter_flux     = cex_erosion_flux(Gamma_CEX, double_ion_ratio, Y_p, Y_pp, sigma_p, sigma_pp)
    dra_dt = sputter_flux * M_grid / (N_A * rho_grid)
    dra_dt = dra_dt*(60)*(60)*(1000) # m / khr
    r_erode = t*dra_dt + ra
    return r_erode

def find_erosion_fit(t):
    ''' Keep for fitting '''
    measured_da_data = pd.read_csv('collision_erosion_model/wirz-time_dependent_fig11.csv', names=['T', 'da'])
    xdata      = measured_da_data['T'].to_numpy()
    ydata      = measured_da_data['da'].to_numpy() / 2 # da -> ra
    popt, pcov = curve_fit(accel_grid_erosion_func, xdata, ydata, p0 = [2.11, 0.39])
    residuals = ydata - accel_grid_erosion_func(xdata, popt[0], popt[1])
    sigma_ab = np.sqrt(np.diagonal(pcov)) # standard deviation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    R2 = 1 - (ss_res / ss_tot)
    print(f"R2 = {R2:0.3}")
    print(f"sigma_ab = {sigma_ab}")
    a = popt[0]
    b = popt[1]

    xfit1   = accel_grid_erosion_func(t, a, b)
    upper_bound = accel_grid_erosion_func(t, a+2*sigma_ab[0], b+2*sigma_ab[1])
    lower_bound = accel_grid_erosion_func(t, a-2*sigma_ab[0], b-2*sigma_ab[1])

    plt.fill_between(t, lower_bound, upper_bound,
                    color = 'black', alpha = 0.15, label=r"$\pm 2\sigma$ bounds")
    plt.plot(t, xfit1, label=f'curve_fit: a={a:0.3}+\-{sigma_ab[0]:0.3f}, b={b:0.3}+\-{sigma_ab[1]:0.3f}, R2={R2:0.3}')

    plt.plot(xdata, ydata, 'k*', label = r"measured NSTAR ELT $r_a$  ")
    #plt.grid(which='both')
    plt.xlabel('Time [khr]')
    plt.ylabel('Accel Grid Radius [mm]')
    #plt.xlim([0,32])
    plt.title('Grid radius over the ELT with ')
    plt.legend()
    #plt.show()
    return a, b, R2, sigma_ab

def accel_grid_erosion_funcV2(t, a, b):
    ''' Keep for curvefitting '''
    dPdt = dfdt(t, a, b)#a*b / (1 + b*t)

    j_bar = 1 #ufloat(1, 0.2)
    #n0_sa = 1e18
    #double_ion_ratio = 0.08 #ufloat(0.08, 0.05)
    lg     = 0.7e-3
    rs     = 1.91/2
    ra     = 1.14/2
    s      = 2.24
    Vd     = 1500
    Va     = -190
    M      = 2.18e-25 # kg, xenon particle mass
    Ec     = 400, #ufloat(400, 200) # using the average of the energy
    E_cex  = 500
    M_grid = 95.95 # g/mol
    rho_grid = 10.22 * (100)**3 # g/cm3
    
    CF = simpleClausingFactor(ra, s)
    ni, n0, y, Te = calculate_parameters_from_clausing(CF)

    jCL              = j_CL(M, Vd-Va, lg)
    sigma_p          = xe_xsections(Ec, "+")*(1e-20) # m^2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20) # m^2
    rb               = ra * beam_area_model(j_bar)**(1/2)
    nominal_beam_vol = beam_volume_trap(lg, rs, rb)
    Vbeam            = nominal_beam_vol
    Gamma_CEX        = dPdt*cex_flux(jCL*j_bar, n0, sigma_p, Vbeam) # Gamma^C_CEX 
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)
    sputter_flux     = cex_erosion_flux(Gamma_CEX, y, Y_p, Y_pp, sigma_p, sigma_pp)
    dra_dt = sputter_flux * M_grid / (N_A * rho_grid)
    dra_dt = dra_dt*(10**3)*(60)*(60)*(1000) # mm / khr
    r_erode = t*dra_dt + ra
    return r_erode

def screen_thickness_erosion(dt, hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
    # Pull arguments
    j_bar, Ibar, y, eta_cex, n_ds           = hyperparams
    Rth, div                                = thruster
    rs, ts, ra, ta, lg, s, M_grid, rho_grid = grids
    ni_dis, n0_dis, Te_dis                  = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C         = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi       = operating_conditions

    # convert special variables to their preferred units
    M_kg    = Mi / ( 1000 * N_A)
    lg_kg   = lg * (1e-3)
    n0_mm3  = n0_dis * (1e-9)

    return 

def accel_radius_erosion_modelV2(dt, hyperparams, thruster, grids, discharge_chamber, facility, operating_conditions):
    # Pull arguments
    j_bar, Ibar, y, eta_cex, n_ds    = hyperparams
    Rth, div          = thruster
    rs, ts, ra, ta, lg, s, M_grid, rho_grid = grids
    ni_dis, n0_dis, Te_dis   = discharge_chamber
    n0_fac, Rch, Lch, Ns, Ms, rho_C = facility
    IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi = operating_conditions

    '''
    j_bar, y                    = hyperparams
    rs, ra, lg, s               = grid_geometry
    M_grid, rho_grid            = grid_mats
    n0, Ec, E_cex, Vd, Va, M    = model_inputs
    '''
    # convert special variables to their preferred units
    M_kg    = Mi / ( 1000 * N_A)
    lg_kg   = lg * (1e-3)
    n0_mm3  = n0_dis * (1e-9)

    jCL              = j_CL(M_kg, Vd-Va, lg_kg)                # -> A m-2 this is overpredicting Ib but it's ok
    jBohm            = bohm_current_density(n0_dis, Te_dis, M_kg)
    jb               = j_bar * jBohm #j_bar * jCL
    sigma_p          = xe_xsections(Ec, "+")*(1e-20)            # -> m2
    sigma_pp         = xe_xsections(Ec, "++")*(1e-20)           # -> m2
    rb               = ra * beam_area_model(j_bar)**(1/2)       # -> mm
    Vbeam            = beam_volume_trap(lg, rs, rb)             # -> mm3
    Gamma_CEX        = cex_flux(jb, n0_mm3, sigma_p, Vbeam)     # -> ions/s, Gamma^C_CEX 
    Y_p              = ey.calculate_yield_Xe_Mo(E_cex)          # -> atoms/ion; produces reasonable results
    Y_pp             = ey.calculate_yield_Xe_Mo(2*E_cex)        # -> atoms/ion; produces reasonable results
    sputter_flux     = cex_erosion_flux(Gamma_CEX, y, Y_p, Y_pp, sigma_p, sigma_pp) # atoms/s
    dra_dt_0         = sputter_flux * M_grid / (N_A * rho_grid) # mm/s
    dra_dt           = dra_dt_0*(60)*(60)*(1000)        # mm / khr
    r_erode          = dt*dra_dt + ra

    return r_erode

def erosion_model():
    # Physical constants and general material properties 
    ## User set values
    # hyperparameters ---------------------------------------
    j_bar    = 0.13     # normalized beamlet current
    I_bar    = 2.1      # beamlet-bohm current ratio
    eta_cex  = 0.01     # cex ion impingement probability
    n_ds     = 10       # thruster radius's downstream 
    # thruster ----------------------------------------------
    lg       = 0.7      # grid separation, mm
    rs       = (1.91)/2 # screen grid radius, mm
    ra0      = (1.14)/2 # Initial accel grid radius, mm
    s        = 2.24     # aperture center spacing, mm
    ta0      = 0.38     # accel grid thickness. mm
    Rth      = 0.15     # thruster major radius, m
    ts       = 0.38     # screen grid thickness, mm
    M_grid   = 95.95    # grid material atomic mass, g/mol
    rho_grid = 0.01022  # grid material density, g/mm3
    diverg   = 20       # thruster divergence angle, degrees
    # facility ----------------------------------------------
    n0_fac   = 2e18     # facility number density, m-3
    Rch      = 2        # chamber radius, m
    Lch      = 3        # chamber length, m
    Ns       = 6        # chamber wall material atomic number
    Ms       = 12.011   # chamber wwall material atomic mass, g/mol
    rho_C    = 2.25e6   # redeposited material density, g/m3
    # model inputs/Operating conditions ---------------------
    IB       = 1.76     # beam current, A
    Ib       = 2.7e-4   # beamlet current, A
    Vd       = 1200     # Discharge potential above cathode, V
    Va       = -190     # Accel potential below cathode, V
    Ni       = 54       # propellant atomic number
    Mi       = 131      # propellant atomic mass, AMU
    Ec       = 400      # Energy at collision, +\-200V
    E_cex    = 400      # Energy of CEX ions colliding with grid, +\-200V
    

    # ELT data
    xdata, ydata = grab_data()

    # initialize results arrays
    dt     = 0.1
    T_end  = 50
    tsteps = np.arange(0, T_end+dt, dt)
    ra_t   = np.zeros(len(tsteps))
    ta_t   = np.zeros(len(tsteps))
    y_arr  = np.zeros(len(tsteps))
    n0_arr = np.zeros(len(tsteps))
    Te_arr = np.zeros(len(tsteps))
    ni_arr = np.zeros(len(tsteps))
    ra_t[0] = ra0
    ta_t[0] = ta0
    i = 1
    for t in tsteps[:-1]:
        # Calculate Clausing Factor with updated accel grid radius
        CF = simpleClausingFactor(ra=ra_t[i-1], s=s)
        # Update discharge chamber plasma parameters
        ni_thstr, n0_thstr, y, Te_thstr = calculate_parameters_from_clausing(CF)
        y_arr[i-1] = y
        n0_arr[i-1] = n0_thstr
        Te_arr[i-1] = Te_thstr
        ni_arr[i-1] = ni_thstr
        # Fill new model inputs 
        hyper = (j_bar, I_bar, y, eta_cex, n_ds)
        thstr = (Rth,  diverg)
        grids = (rs, ts, ra_t[i-1], ta_t[i-1], lg, s, M_grid, rho_grid)
        disch = (ni_thstr, n0_thstr, Te_thstr)
        fclty = (n0_fac, Rch, Lch, Ns, Ms, rho_C)
        opcon = (IB, Ib, Ec, E_cex, Vd, Va, Ni, Mi)
        # Calculate new accel grid radius 
        ra_t[i] = accel_radius_erosion_modelV2(dt, hyper, thstr, grids, disch, fclty, opcon) 
        ta_t[i] = accel_thickness_erosion_modelV2(dt, hyper, thstr, grids, disch, fclty, opcon)
        i+=1
    
    #y_arr[i] = y
    #n0_arr[i] = ni_thstr
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    ax = axs[0]
    ax.plot(tsteps, ra_t, 'r', label = "Clausing Factor Modified Params")
    ax.plot(xdata, ydata, 'k*', label = 'ELT Data')
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$r_a [mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[1]
    ax.plot(tsteps, ta_t, 'r', label = "Clausing Factor Modified Params")
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$t_a [mm]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    #plt.show()

    until = -1
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12,8))
    ax = axs[0][0]
    ax.plot(tsteps[:until], y_arr[:until], 'r', label = "Clausing Factor Modified Params")
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$\gamma$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[0][1]
    ax.plot(tsteps[:until], n0_arr[:until], 'r', label = "Clausing Factor Modified Params")
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$n_0[m^{-3}]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[1][0]
    ax.plot(tsteps[:until], Te_arr[:until], 'r', label = "Clausing Factor Modified Params")
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$T_e[eV]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    ax = axs[1][1]
    ax.plot(tsteps[:until], ni_arr[:until], 'r', label = "Clausing Factor Modified Params")
    ax.set_xlabel(r'$t [khr]$', fontsize=18)
    ax.set_ylabel(r'$n_i[m^{-3}]$', fontsize=18)
    ax.grid(which='both')
    ax.legend()
    plt.show()

    return ra_t


def vol_test():
    r = 1
    R = 1
    h = 3
    #print(f"{beam_volume_trap(h, r, r): 0.3f}")
    #print(f"{plume_volume(r, h, 0):0.3f}")
    V1 = beam_volume_trap(h, r, r)
    V2 = plume_volume(r, h, 0)
    return V1==V2
        
def main():
    #find_erosion_fit(np.linspace(0,35))
    erosion_model()
    #print("---")
    #iter_thickness_test()
    #print("---")
    #ta_dot = sputterant_thickness_rate(1.76, 2, 6, 54, 12.01, 131.1, 2.25*(100**3), 1100)
    #print(f"{ta_dot*60*60} m/hr")
    #print(f"{ta_dot*1000*1000*60*60*1000} um/khr")

    #s_avg = sputterant_redep_rate(1.76, 2, 0.3, 6, 54, 12.01, 131.1, 2.25*(100**3), 1100)
    #print(s_avg)
    #I_cex = cex_generation_rate(1.76, 0.3, 2e18, xe_xsections(1100))
    #print(I_cex)
    #thetaC = carbon_surface_coverageV2(s_avg, 0.01*I_cex)
    #print(thetaC)
    
    
    return





if __name__ == "__main__":
    main()
    #iter_erosion_model()
    #iter_thickness_model()
