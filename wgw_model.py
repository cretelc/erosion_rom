from numpy import array, arctan, sqrt, exp, log, linspace
from scipy.constants import pi
from scipy.constants import epsilon_0 as eps0
from scipy.constants import elementary_charge as q
from scipy.constants import electron_mass as me
from scipy.constants import Avogadro as N_A
import matplotlib.pyplot as plt
import pandas as pd

## Functions
def AMU2kg(Ma):
    return (1e-3) * Ma / N_A

def f(x, a, b, c):
    return a*x**2 + b*x + c

def f_sqrt(x, a, b):
    return a*x**(0.5) + b

def saddlepoint_Voltage(Va, da, Vdp, le, ta, dV=0):
    ''' Equation 9 or when dV=0, Equation 7, Williams 2003'''
    A = da*(Vdp - Va)/(2 * pi * le)
    B = (2*ta/da) * arctan(da/(2*ta))
    C = exp(-ta/da)
    Vsp = Va + dV + A*(1 - B)*C
    return Vsp

def spacecharge_effect(Jb, Vdp, Vsp, ion_mass, db, da):
    ''' Equations 8, Williams 2003'''

    M = AMU2kg(ion_mass)
    A = Jb / (2 * pi * eps0)
    B = (M/(2 * q *(Vdp - Vsp)))**(1/2)
    C = (1/2) - log(db/da)

    dV = A*B*C
    return dV

def eff_grid_gap(lg, ts, ds):
    ''' NOT Williams 2003, need to find this'''
    return ((lg + ts)**2 + (ds/2)**2)**(1/2)

def B_term(da, le, ta):
    ''' Williams 2003'''
    A = da/(2 *pi *le)
    B = (2*ta/da)*arctan(da/(2*ta))
    C = exp(-ta/da)
    return A * (1 - B) * C

def retarding_Va(Vsp, dV, Vdp, B):
    ''' Equation 10, Williams 2003  '''
    Va = (Vsp - dV - B*Vdp) / (1-B)
    return Va

def retarding_Vsp(Vbp, Te, ebs_ratio, ion_mass, Vdp):
    ''' Equation 5, Williams 2003'''
    M = AMU2kg(ion_mass)
    A = 2*ebs_ratio
    B = pi*(me/M)
    C = (Vdp-Vbp) / Te
    return Vbp + Te*log(A*sqrt(B*C))

def beamlet_to_ebs_ratio(Vbp, Vdp, Vsp, Te, Ma):
    M = AMU2kg(Ma)
    A = pi * M/me
    #print(A)
    B = Te / (Vdp - Vbp)
    #print(B)
    C = exp(-(Vbp - Vsp) / Te)
    #print(f"C = {C}")
    R = (1/2) * (A*B)**(1/2) * C
    return R

def grab_wgw_data():
    ABC = ['a', 'b', 'c']
    data_dic = {}
    for letter in ABC:
        file = f"fig6{letter}"
        fig6_wgw_data = pd.read_csv(f'erosion_rom/{file}.csv', names=['JB', 'Vebs'])
        data_dic[file] = [fig6_wgw_data['JB'].to_numpy(), fig6_wgw_data['Vebs'].to_numpy()]
        #ydata      = measured_da_data['da'].to_numpy() / 2 # da -> ra
    return data_dic

def electron_backstreaming(beam_potential, 
                           discharge_potential, 
                           accel_voltage,
                           beam_current_density,  
                           beam_Te, rs, ra, ts, ta, lg, 
                           rb, Ma, ebs_ratio):
    ''' runs the electron backstreaming model '''

    ''' Unique (input) variables 
    Vbp - beam plasma potential, V
    Vdp - discharge plasma potential
    Va  - accel grid potential 
    Jb  - beam current density
    Te  - beam electron temperature, eV 
    ra  - accel radius, m 
    le  - effective grid separation 
    ta  - accel thickness
    db  - beam diameter 
    Ma  - atomic mass
    '''
    Vbp = beam_potential
    Vdp = discharge_potential
    Va  = accel_voltage
    Jb = beam_current_density
    Te = beam_Te
    da = 2*ra 
    ds = 2*rs
    db = 2*rb
    le = eff_grid_gap(lg, ts, ds)
    #ebs_ratio = 1e-3
    ## Electron backstreaming model ---
    # Initial Calculations
    Vsp_star = retarding_Vsp(Vbp, Te, ebs_ratio, Ma, Vdp) 
    B        = B_term(da, le, ta)

    # Calculate unperturbed saddlepoint potential (Equation 7)
    Vsp = saddlepoint_Voltage(Va, da, Vdp, le, ta)

    # Calculate the space charge perturbation (Equation 8)
    dV     = spacecharge_effect(Jb, Vdp, Vsp, Ma, db, da)
    
    Va_ebs = retarding_Va(Vsp_star, dV, Vdp, B)
    return Va_ebs

def williams_test():
    da = 4.5 * (1e-3) # m 
    ta = 4.0 * (1e-3) # m
    ds = 9.0 * (1e-3) # m
    ts = 1.5 * (1e-3) # m
    lg = linspace(7, 10) * (1e-3) 
    lc = 10.4 * (1e-3) # m
    Vdp = 13.03 * (1e3) # V
    Vbp = 0 # V

    # Variables 
    JB = linspace(1, 2) 

if __name__ == '__main__':
    # Input Parameters from Table 1 
    # Accel Grid Potential, -V
    Va = -180        
    # Discahrge Potential, V
    Vdp = 13030      
    # Beam Potential, V
    Vbp = 0        
    # Beamlet current, A
    Jb = [(1e-3) * linspace(1.0, 4.0),
          (1e-3) * linspace(1.0, 4.0),
          (1e-3) * linspace(1.0, 4.0)]  
     # Atomic Mass, AMU  
    Ma = 84       
    # Grid separation, m
    lg_vec = (1e-3)*array([7, 8.5, 10])
  
    # Accel Grid thickness, m  
    ta = (1e-3)*4.0    
    # Accel Grid Diameter, m 
    da =(1e-3) * 4.5  
    # Screen Grid Diameter, m
    ds = (1e-3) * 9
    # Screen Grid Thickness, m
    ts = (1e-3) * 1.5  
    # Temperature, eV
    Te = 1       
    # Backstreaming-to-Beamlet Current Ratio
    R = 1e-3 

    # Coefficients for modeling (db/da = f(jb)) 
    coeffs = [[0.190, -0.895, 1.486],
              [0.400, -1.560, 1.920],
              [0.55, -1.895, 2.047]]
    
    fig, axs = plt.subplots(1, 1)
    ABC = ['c', 'b', 'a']
    color = ['r', 'g', 'b']
    wgw_dic = grab_wgw_data()
    for i, lg in enumerate(lg_vec):
        print(f'\nlg = {lg*1000}')
        le = eff_grid_gap(lg, ts, ds)
        a = coeffs[i][0]
        b = coeffs[i][1]
        c = coeffs[i][2]
        db_da = f(Jb[i]*1000, a, b, c)
        # Beamlet diameter
        db = db_da * da

        ## Electron backstreaming model ---
        # Initial Calculations
        Vsp_star = retarding_Vsp(Vbp, Te, R, Ma, Vdp) 
        B        = B_term(da, le, ta)

        # Calculate unperturbed saddlepoint potential (Equation 7)
        Vsp = saddlepoint_Voltage(Va, da, Vdp, le, ta)

        # Calculate the space charge perturbation (Equation 8)
        dV     = spacecharge_effect(Jb[i], Vdp, Vsp, Ma, db, da)
        
        Va_ebs = retarding_Va(Vsp_star, dV, Vdp, B)

        # Plots
        l = f"lg = {lg*1000:0.1f} mm"
        axs.plot(Jb[i]*1000, -Va_ebs, color=color[i], label = l)
        run = f'fig6{ABC[i]}'
        axs.plot(wgw_dic[run][0],wgw_dic[run][1], color=color[i], linewidth=0, marker='*', label=f'wgw_data {run}')
        #axs[1].semilogy(Jb[i]*1000, R_ebs, label = f"lg = {lg*1000:0.1f} mm")
        #axs[2].plot(Jb[i]*1000, dV, label = f"lg = {lg*1000:0.1f} mm")
    
    #wgw_dic = grab_wgw_data()
    #f = 'fig6a'
    #axs.plot(wgw_dic[f][0],wgw_dic[f][1], 'r*')
    #f = 'fig6b'
    #axs.plot(wgw_dic[f][0],wgw_dic[f][1], 'g*')
    #f = 'fig6c'
    #axs.plot(wgw_dic[f][0],wgw_dic[f][1], 'b*')
    axs.set_xlabel('Beamlet Current [mA]')
    axs.legend()
    axs.set_ylabel(r'$V_{ebs}$')
    axs.grid(which='both')
    axs.set_ylim([100,500])
    plt.show()

