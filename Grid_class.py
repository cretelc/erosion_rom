import numpy as np


class Grid():
    def __init__(self, rs, ts, ra, ta, lg, accel_material, grid_material):
        return 
    

    def erode():
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
