# third party imports
import os
import juliapkg

# MUST be before importing julia
os.environ["JULIA_NUM_THREADS"] = "1"
juliapkg.add("DifferentialEquations", "0c46a032-eb83-5123-abaf-570d42b7fbaa")
juliapkg.add("Sundials", "c3572dad-4567-51f8-b174-8c6c989267f4")
juliapkg.resolve()

from juliacall import Main as de
de.seval("using DifferentialEquations, Sundials")

from pathlib import Path
import numpy as np
import pandas as pd
import time
import gc

# the following lines ensure that we can import local modules with a path from the project root directory
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
#print("Python Path:", sys.path)
#print("Script Location:", os.path.abspath(__file__))
#print("Current Working Directory:", os.getcwd())

# local imports
from model.blood_pressure.hormone_influence import HormoneInfluence, HormoneInfluenceContraceptive
from model.blood_pressure.file_handling import HUMAN_DATA_DIR_CSV, HUMAN_SIMULATION_DIR, HUMAN_EVALUATION_DIR, get_simulation_parameter_values_filename, load_initial_values, load_file_contents_by_column, load_df_from_file, get_initial_values_filename, get_simulation_result_filename, get_model_parameters_filename, load_hormone_influence_on_parameters, save_simulation_results, load_file_contents, get_cycle_phase_definition_filename, get_evaluation_average_filename, get_evaluation_difference_filename, get_evaluation_phase_average_filename, get_species_indices, get_species_names_and_indices
from model.blood_pressure.visualisation import create_simulation_results_visualisation_hormone_influence, create_MAP_comparison_visualisation_hormone_influence, create_evaluation_result_plots, create_simulation_average_plots, create_simulation_average_plots_per_cycle_phase, downsample_df

# Based on the publication
# Jessica Leete and Anita T. Layton. Sex-specific long-term blood pressure regulation: Modeling
# and analysis. Computers in Biology and Medicine, 104:139–148, January 2019. ISSN 1879-0534. doi: 10.1016/j.compbiomed.2018.11.002

# This code is a translation of the original MATLAB code into Python.
# sodium -> Na
# potassium -> K

def bp_model(residual, dy, y, p, t):
    '''
    This is the function containing the computational model of sex-specific long-term blood pressure regulation. The respective 
    parameterization is established in parameter p: p = (female,) + (hormone_influence,) + tuple(model_parameters).
    More information about the configuration in function 'run_simulation()'.
    '''
    female = p[0]
    hormone_influence = p[1]
    model_parameters = p[2:]

    # Phi_sodin_const is only used in the rat model and thus not included in the human model parameters
    # apply hormone influence on model parameters
    if hormone_influence.duration > 0:
        if female:
            model_parameters = hormone_influence.apply_hormone_influence_on_parameters(model_parameters, t)
 
    N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters

    # unpack the ODE variables
    ode_indices = get_species_indices('ODE')
    V_ecf, vas, a_baro, N_adh, delta_ra, M_sod, N_al, AGT, PRC, AngI, AngII, AT1R, AT2R, Ang17, AngIV = [y[i] for i in ode_indices]

    # unpack the algebraic variables
    alg_indices = get_species_indices('algebraic')
    rsna, alpha_map, alpha_rap, R_r, beta_rsna, Phi_rb, Phi_gfilt, P_f, P_gh, Sigma_tgf, Phi_filsod, Phi_ptsodreab, eta_ptsodreab, gamma_filsod, gamma_at, gamma_rsna, Phi_mdsod, Phi_dtsodreab, eta_dtsodreab, psi_al, Phi_dtsod, Phi_cdsodreab, eta_cdsodreab, lambda_dt, lambda_anp, lambda_al, Phi_usod, Phi_sodin, V_b, P_mf, Phi_vr, Phi_co, P_ra, vas_f, vas_d, R_a, R_ba, R_vr, R_tp, P_ma, epsilon_aum, a_auto, a_chemo, C_adh, N_adhs, C_sod, nu_mdsod, nu_rsna, C_al, N_als, xi_ksod, xi_map, xi_at, hatC_anp, nu_AT1, R_sec, PRA, R_aa, R_ea, Sigma_myo, Psi_AT1RAA, Psi_AT1REA, Psi_AT2RAA, Psi_AT2REA, Phi_twreab, mu_adh, mu_Na, Phi_u, Phi_win = [y[i] for i in alg_indices]

    ## Default parameter inputs for changes in simulation.
    # Drugs
    NSAID           = 0 # NSAID indicator. O for none, 1 for normal, 2 for high dose.

    # Renal perfusion pressure
    RPP_ind = False # Boolean for controlling renal perfusion pressure

    # Autoregulatory mechanisms
    denerve     = False # Boolean for unilateral renal denervation by fixing rsna = 1, which is baseline.
    no_TGF      = False # Boolean for canceling the tubuloglomerular feedback.
    no_Myo      = False # Boolean for canceling the myogenic response.
    lin_Myo     = False # Boolean for having linear myogenic response.
    imp_Myo_ind = False # Boolean for having impaired myogenic response.

    # Water intake
    low_win_ind = False # Boolean for having low water intake.

    # Sex specific mechanisms
    m_RSNA = False # Boolean for having male RSNA in the female model.
    m_AT2R = False # Boolean for having male effect of AT2R in the female model.  

    tchange = 100
    # Renal perfusion pressure
    if RPP_ind:
        if t <  tchange:
            RPP = 0 # TODO: SSdata_fix[41]
        elif t >= tchange: 
            RPP = 0 # TODO: RPP_per * np.tanh(5 * (t-tchange)) + SSdata_fix[41]
        
    # Drugs
    # deltat = 30 # minutes over which to continuously increase drug dose
    #if t < tchange:
    kappa_ACEi  = 0
    kappa_f     = 0
    kappa_f_md  = 0 
    kappa_AngII = 0 
    kappa_ARB   = 0
    
    #### Optional Parameters ####

    # rsna: renal sympathetic nerve activity
    rsna0 = N_rsna * alpha_map * alpha_rap
    if not female:
        residual[0] = rsna - rsna0
    elif female:
        if m_RSNA:
            residual[0] = rsna - rsna0
        else:
            residual[0] = rsna - (rsna0**(1/rsna0)) # less excitable and more easily repressed RSNA in females
        
    # alpha_map: effect of mean arterial pressure on renal sympathetic nerve activity
    alphamap_a = 100
    residual[1] = alpha_map - (0.5 + 1 / (1 + np.exp((P_ma - alphamap_a) / 15)))

    # alpha_rap: effect of right atrial pressure on renal sympathetic nerve activity
    residual[2] = alpha_rap - (1 - 0.008 * P_ra)

    # R_r: renal vascular resistance
    residual[3] = R_r - (R_aa + R_ea)

    # beta_rsna: Effect of renal sympathetic nerve activity on afferent arteriolar resistance
    if denerve:
        residual[4] = beta_rsna # TODO: - SSdata_fix[4]
    else:
        residual[4] = beta_rsna - (2 / (1 + np.exp(-3.16 * (rsna - 1))))

    # Phi_rb: renal blood flow
    if RPP_ind:
        residual[5] = Phi_rb - (RPP / R_r)
    else:
        residual[5] = Phi_rb - (P_ma / R_r)

    # Phi_gfilt: glomerular filtration rate
    # C_gcf: glomerular capillary filtration coefficient
    residual[6] = Phi_gfilt - (P_f * C_gcf) 

    # P_f: net filtration pressure
    residual[7] = P_f - (P_gh - (P_B + P_go))
    # residual[7] = (59.7-18-28)/59.7 * P_gh

    # P_gh: glomerular hydrostatic pressure
    if RPP_ind:
        residual[8] = P_gh - (RPP - Phi_rb * R_aa)
    else:
        residual[8] = P_gh - (P_ma - Phi_rb * R_aa)

    # Sigma_tgf: tubuloglomerular feedback signal
    Sigmatgf_a = 3.859

    # sigma_tgf
    if no_TGF:
        residual[9] = Sigma_tgf - 1
    elif NSAID == 1:
        residual[9] = Sigma_tgf - (0.644032 + 1.188073289 / (2.0285174154 + np.exp(((1-kappa_f_md) * Phi_mdsod - 3.859)/(-0.9617))))
    elif NSAID > 11:
        residual[9] = Sigma_tgf - 1
    else:  # normal TGF # Macula densa
        residual[9] = Sigma_tgf - (0.3408 + (3.449 / (3.88 + np.exp(((1-kappa_f_md) * Phi_mdsod - Sigmatgf_a) / (-0.9617) ))) ) 

    # Phi_filsod: filtered sodium load
    residual[10] = Phi_filsod - (Phi_gfilt * C_sod)

    # Phi_ptsodreab: absolute proximal tubular sodium reabsorption rate
    residual[11] = Phi_ptsodreab - (Phi_filsod * eta_ptsodreab)

    # eta_ptsodreab: fractional proximal sodium reabsorption
    residual[12] = eta_ptsodreab - ((1-kappa_f)* eta_ptsodreab_eq * gamma_filsod * gamma_at * gamma_rsna)
    # residual[12] = (1-0.1*(scale_sodin-1)) * (1-kappa_f)* eta_ptsodreab_eq * gamma_filsod * gamma_at * gamma_rsna

    # gamma_filsod: effect of the filetered sodium load on fractional proximal sodium reabsorption
    gammafilsod_a = 0.85
    gammafilsod_b = 18
    residual[13] = gamma_filsod - (gammafilsod_a + 0.3 / (1 + np.exp((Phi_filsod - gammafilsod_b)/(138) )))

    # gamma_at: effect of angiotensin hormone concentration on fractional proximal sodium reabsorption
    gammaat_a = 0.12  
    gammaat_b = 2.3418
    gammaat_c = 2.6/2.3418
    gammaat_d = 0.95 
    residual[14] = gamma_at - (gammaat_d + gammaat_a / (1 + np.exp(-gammaat_b * ((AT1R/AT1R_eq) - gammaat_c)) ))
    # [AT1R,AT1R_eq, AT1R/AT1R_eq,gammaat_c,gamma_at]

    # gamma_rsna: effect of renal sympathetic nerve activity on fractional proximal sodium reabsorption
    if denerve:
        residual[15] = gamma_rsna # TODO: - SSdata_fix[15]
    else:
        residual[15] = gamma_rsna - (0.72 + 0.56 / (1 + np.exp((1 - rsna) / 2.18)))

    # Phi_mdsod: macula densa sodium flow
    residual[16] = Phi_mdsod - (Phi_filsod - Phi_ptsodreab)

    # Phi_dtsodreab: absolute distal tubule sodium reabsorption rate
    residual[17] = Phi_dtsodreab - (Phi_mdsod * eta_dtsodreab)

    # eta_dtsodreab: fractional distal tubule sodium reabsorption
    residual[18] = eta_dtsodreab - (eta_dtsodreab_eq * psi_al)

    # psi_al: effect of aldosterone concentration on fractional distal sodium reabsorption
    # including competitive binding of mineralocorticoid receptor (MR) activity by progesterone (P4)
    residual[19] = psi_al - (0.17 + 0.94/(1+np.exp((0.48 - 1.2*(np.log(C_al * gamma_MR)))/0.88)))
    
    # Phi_dtsod: distal tubule sodium outflow
    residual[20] = Phi_dtsod - (Phi_mdsod - Phi_dtsodreab)

    # Phi_cdsodreab: absolute collection duct sodium reabsorption rate
    residual[21] = Phi_cdsodreab - (Phi_dtsod * eta_cdsodreab)

    # eta_cdsodreab: fractional collection duct sodium reabsorption
    eta_cdsodreab0 = ( eta_cdsodreab_eq * lambda_dt * lambda_anp * lambda_al)
    residual[22] = eta_cdsodreab - eta_cdsodreab0
    
    # lambda_dt: effect of distal tubule sodium outflow on collection duct sodium reabsorption rate
    lambdadt_a = 0.82    
    lambdadt_b = 0.39         
    lambdadt_c = 1/0.375 
    lambdadt_d = 1.7625
    residual[23] = lambda_dt - (lambdadt_a + lambdadt_b/ (1 + np.exp(lambdadt_c* (Phi_dtsod - lambdadt_d)) ))

    # lambda_anp: effect of atrial natriuretic peptide on collecting duct sodium reabsorption rate
    residual[24] = lambda_anp - (-0.1 * hatC_anp + 1.1)
    
    # lambda_al: effect of ALD on collecting duct K+ secretion
    lambdaal_a = 1/0.76
    residual[25] = lambda_al - (1/lambdaal_a * C_al**0.06)
    
    # Phi_usod: urine sodium flow
    # softplus
    residual[26] = Phi_usod - np.log(1 + np.exp( Phi_dtsod - Phi_cdsodreab ))
    #residual[26] = Phi_usod - max(0,( Phi_dtsod - Phi_cdsodreab ))


    # Phi_sodin: sodium intake
    sodin_E = 0.14 
    sodin_C = 0.14 
    sodin_D = 0.765 
    sodin_L = 0.1
    sodin_B = (sodin_L*sodin_C - sodin_E*sodin_C +(sodin_E - 0.126)*sodin_C*ALD_eq**sodin_D)/(0.126-sodin_L)
    sodin_A = (0.126-sodin_E)*(sodin_B+sodin_C*ALD_eq**sodin_D)

    # softplus
    residual[27] = Phi_sodin - np.log(1 + np.exp(sodin_A/(sodin_B+sodin_C*C_al**sodin_D)+sodin_E))
    #residual[27] = Phi_sodin - max(0,sodin_A/(sodin_B+sodin_C*C_al**sodin_D)+sodin_E)

    # V_ecf, ODE: extracellular fluid volume
    # residual[28] = V_ecf_p - ( Phi_win - Phi_u )
    residual[28] = dy[28] - ( Phi_win - Phi_u )

    # V_b: blood volume
    # the equations for male and female are equal, taken from baseline implementation
    if female:
        residual[29] = V_b - (0.325 * V_ecf)
    else:
        residual[29] = V_b - (0.325 * V_ecf)
    
    # P_mf: mean filling pressure
    if female:
        pmf_a = 1.1*(7.4360)
    else:
        pmf_a = (7.4360)
    residual[30] = P_mf - (( pmf_a * V_b - 30.18) * epsilon_aum)

    # Phi_vr: venous return
    residual[31] = Phi_vr - ((P_mf - P_ra) / R_vr)

    # Phi_co: cardiac output
    # in liters per minute, clinical equivalent of blood flow
    # blood flow is defined by Poiseuille's law as Q = delta_P * (PI*r^4)/(8*NL),
    # with Q bein blood flow, delta_P the pressure gradient (Mean arterial pressure), r as the radius
    # of the blood vessel, N as the viscosity of the blood, L as the length of the vessel.
    # This formula is commonly restated as Phi_co = P_ma / TPR
    residual[32] = Phi_co - Phi_vr

    # P_ra: right atrial pressure: representation of the heart pump function
    if female:
        pra_a = 0 #0.8245
    else:
        pra_a = 0 #0.8268
    residual[33] = P_ra - (0.2787 * np.exp(Phi_co * 0.2281) - pra_a)
          
    # vas, ODE: vascularity
    # residual[34] = vas_p - ( 1 / 1000 * (vas_f - vas_d) )
    residual[34] = dy[34] - ( 1 / 1000 * (vas_f - vas_d) )

    # vas_f: vascularity formation rate
    if female:
        vasf_a = 0.4799 #*1.15
    else:
        vasf_a = 0.4799
    residual[35] = vas_f - ((11.312 * np.exp(-Phi_co * vasf_a)) / 100)
    # residual[35], [vas_f, Phi_co, vasf_a], 

    # vas_d: vascularity destruction rate
    residual[36] = vas_d - (vas * K_vd)
    # [residual[36], vas_d, vas, K_vd]

    # R_a: arterial resistance
    # This is the equation where we want to include the effect of estrogen and progesterone on arterial resistance via a submodel        
    residual[37] = R_a - (R_ba * epsilon_aum)

    # R_ba: basic arterial resistance
    residual[38] = R_ba - ((K_bar) / vas)

    # R_vr: resistance to venous return
    residual[39] = R_vr - ((8 * R_bv + R_a) / 31)

    # R_tp: total peripheral resistance
    residual[40] = R_tp - (R_a + R_bv)

    # P_ma: mean arterial pressure
    residual[41] = P_ma - (Phi_co * R_tp)

    # epsilon_aum: autonomic multiplier effect
    residual[42] = epsilon_aum - (a_chemo + a_baro)

    # a_auto: autonomous system activity
    aauto_a = 0.011
    residual[43] = a_auto - (3.0042 * np.exp(-aauto_a * P_ma))

    # a_chemo: chemoreceptor activity
    residual[44] = a_chemo - (1/4 * a_auto)

    # a_baro, a_auto, ODE, a_baro: baroreceptor activity
    # residual[45] = a_baro_p - ( 3/4 * (a_auto_p - 0.0000667 * (a_baro - 1)) )
    residual[45] = dy[45] - ( 3/4 * ((dy[43]) - 0.0000667 * (a_baro - 1)) )

    # C_adh: antidiuretic hormone concentration
    residual[46] = C_adh - (4 * N_adh) 

    # N_adh, ODE: normalized antidiuretic hormone concentration
    # residual[47] = N_adh_p - ( 1/T_adh * (N_adhs - N_adh) )
    residual[47] = dy[47] - ( 1/T_adh * (N_adhs - N_adh) )

    # N_adhs: normalized antidiuretic hormone secretion rate
    Nadhs_a = 141
    # apply softplus to ensure that the secretion rate N_adhs is non-negative
    N_adhs_tmp = N_adhs_eq * (np.log(1 + np.exp(C_sod - Nadhs_a)) + np.log( 1 + np.exp(epsilon_aum - 1 )))
    residual[48] = N_adhs - np.log(1 + np.exp((N_adhs_tmp - (N_adhs_eq * delta_ra)) / 3))
    # softplus
    #residual[48] = N_adhs - (N_adhs_eq * (max( 0, C_sod - Nadhs_a) + max( 0, epsilon_aum - 1 ) - delta_ra) / 3)

    # delta_ra, ODE: effect of right atrial pressure on normalized antidiuretic hormone secretion rate
    # residual[49] = delta_ra_p - ( 0.2 * P_ra_p - 0.0007 * delta_ra )
    residual[49] = dy[49] - (0.2 * dy[33]  - 0.0007 * delta_ra)

    # M_sod, ODE: total amount of sodium
    # residual[50] = M_sod_p - ( Phi_sodin - Phi_usod )
    residual[50] = dy[50] - ( Phi_sodin - Phi_usod )

    # C_sod: sodium concentration
    residual[51] = C_sod - (M_sod / V_ecf)

    # nu_mdsod: effect of macula densa soduim flow on normalized renin secretion rate
    if female:
        numdsod_a = 1.637
    else:
        numdsod_a = 1.731

    if NSAID > 0:
        residual[52] = nu_mdsod - (0.2262 + 83.4095/ (77.6196 + np.exp(((1-kappa_f_md) * Phi_mdsod - numdsod_a) / (0.6056) )))
    else:
        residual[52] = nu_mdsod - (0.2262 + 28.04  / (11.56   + np.exp(((1-kappa_f_md) * Phi_mdsod - numdsod_a) / (0.6056) )))

    # nu_rsna: effect of renal sympathetic nerve activity on normalized renin secretion rate
    nursna_a = 0.8667
    if denerve:
        residual[53] = nu_rsna # TODO: - SSdata_fix[53]
    else:
        residual[53] = nu_rsna - (1.822 - 2.056 / (1.358 + np.exp(rsna - nursna_a)))

    # C_al: aldosterone concentration
    # softplus
    residual[54] = C_al - (1 + np.log(1 +  np.exp(N_al * ALD_eq - 1)))
    #residual[54] = C_al - max( 1, N_al * ALD_eq )


    # N_al, ODE: normalized aldosterone concentration
    # residual[55] = N_al_p - ( 1/T_al * (N_als - N_al) )
    residual[55] = dy[55] - ( 1/T_al * (N_als - N_al) )

    # N_als: normalized aldosterone secretion rate
    residual[56] = N_als - (N_als_eq * xi_ksod * xi_map * xi_at)
    
    # xi_ksod: effect of potassium to sodium concentration ratio on normalized aldosterone secretion rate
    C_K_ref = 4.2
    xi_ksod_par = 6
    # softplus is NOT used here since it causes the system to oscillate
    #residual[57] = xi_ksod - np.log(1 + np.exp((C_K / C_sod) / (C_K_ref/144/(xi_ksod_par+1)) - xi_ksod_par ))
    residual[57] = xi_ksod - max( 0, (C_K / C_sod) / (C_K_ref/144/(xi_ksod_par+1)) - xi_ksod_par )

    
    # xi_map: effect of mean arterial pressure on normalized aldosterone secretion rate
    if P_ma <= 100:
        residual[58] = xi_map - ((1/np.exp(-0.0425 * 100)) * np.exp(-0.0425 * P_ma))
    else:
        residual[58] = xi_map - 1

    # xi_at: effect of angiotensin hormone on normalized aldosterone secretion rate
    # the values are the same for female and male, taken from baseline implementation
    xiat_a = 0.47
    xiat_b = 2.4 
    xiat_c = 1.5*1.301/0.8
    xiat_d = 2.82/0.8/xiat_c
    if female:
        xiat_d = xiat_d - 0.4
    
    residual[59] = xi_at - (xiat_a + xiat_b / (1 + np.exp(-xiat_c * ((AT1R/AT1R_eq) - xiat_d)) ))
 
    # hatC_anp: normalized atrial natriuretic peptide concentration
    residual[60] = hatC_anp - (7.4052 - 6.554 / (1 + np.exp(P_ra - 3.762)))

    # AGT, ODE: Angiotensinogen, reaction cascade from AGT to Ang (1-7)/ Ang II/ Ang IV
    # residual[61] = AGT_p - ( k_AGT - PRA - np.log(2)/h_AGT * AGT )
    residual[61] = dy[61] - ( k_AGT - PRA - np.log(2)/h_AGT * AGT )

    # nu_AT1: Angiotensin II type 1
    residual[62] = nu_AT1 - (10**(0.0102 - 0.95 * np.log10(AT1R / AT1R_eq)))
    
    # R_sec: renin secretion rate
    residual[63] = R_sec - (N_rs * nu_mdsod * nu_rsna * nu_AT1)

    # PRC, ODE: plasma renin concentration
    # residual[64] = PRC_p - ( R_sec - np.log(2)/h_renin * PRC )
    residual[64] = dy[64] - ( R_sec - np.log(2)/h_renin * PRC )

    # PRA: plasma renin activity
    residual[65] = PRA - (PRC * X_PRCPRA)

    # AngI, ODE: Angiotensin I,part of the reaction cascade AGT -> Ang 
    # residual[66] = AngI_p - ( PRA - ((1-kappa_ACEi) * c_ACE + c_Chym + c_NEP) * AngI - np.log(2)/h_AngI * AngI )
    residual[66] = dy[66] - ( PRA - ((1-kappa_ACEi) * c_ACE + c_Chym + c_NEP) * AngI - np.log(2)/h_AngI * AngI )

    # AngII, ODE: Angiotensin II, part of the reaction cascade AGT -> Ang
    # residual[67] = AngII_p - ( kappa_AngII + ((1-kappa_ACEi) * c_ACE + c_Chym) * AngI - (c_ACE2 + c_IIIV + (1-kappa_ARB) * c_AT1R + c_AT2R) * AngII - np.log(2)/h_AngII * AngII )
    residual[67] = dy[67] - ( kappa_AngII + ((1-kappa_ACEi) * c_ACE + c_Chym) * AngI - (c_ACE2 + c_IIIV + (1-kappa_ARB) * c_AT1R + c_AT2R) * AngII - np.log(2)/h_AngII * AngII )
    
    # AT1R, ODE: Angiotensin II type 1 receptor
    # residual[68] = AT1R_p - ( (1-kappa_ARB) * c_AT1R * AngII - np.log(2)/h_AT1R * AT1R )
    residual[68] = dy[68] - ( (1-kappa_ARB) * c_AT1R * AngII - np.log(2)/h_AT1R * AT1R )
    
    # AT2R, ODE: Angiotensin II type 2 receptor
    # residual[69] = AT2R_p - ( c_AT2R * AngII - np.log(2)/h_AT2R * AT2R )
    residual[69] = dy[69] - ( c_AT2R * AngII - np.log(2)/h_AT2R * AT2R )
    
    # Ang17, ODE: angiotensin (1-7)
    # residual[70] = Ang17_p - ( c_NEP * AngI + c_ACE2 * AngII - np.log(2)/h_Ang17 * Ang17 )
    residual[70] = dy[70] - ( c_NEP * AngI + c_ACE2 * AngII - np.log(2)/h_Ang17 * Ang17 )
    
    # AngIV, ODE: Angiotensin IV
    # residual[71] = AngIV_p - ( c_IIIV * AngII - np.log(2)/h_AngIV * AngIV )
    residual[71] = dy[71] - ( c_IIIV * AngII - np.log(2)/h_AngIV * AngIV )
    
    # R_aa: afferent arteriolar resistance
    residual[72] = R_aa - (R_aass * beta_rsna * Sigma_tgf * Sigma_myo * Psi_AT1RAA * Psi_AT2RAA)
    ## FFFFF
    # residual[72] = R_aass * 0.9130 * Sigma_tgf

    # R_ea: efferent arteriolar resistance
    residual[73] = R_ea - (R_eass * Psi_AT1REA * Psi_AT2REA)

    # Sigma_myo: myogenic effect
    if imp_Myo_ind:
        sigmamyo_a = 1.2 
        sigmamyo_b = 0.3
    else:
        sigmamyo_a = 0.9 
        sigmamyo_b = 1.0

    sigmamyo_d = 0.9
    sigmamyo_c = 9
    sigmamyo_e = 62

    if no_Myo:
        residual[74] = Sigma_myo - 1
    elif lin_Myo:
        residual[74] = Sigma_myo - (5 * (P_gh / sigmamyo_e - 1) + 1)
    else:
        residual[74] = Sigma_myo - (sigmamyo_a + sigmamyo_b / ( 1 + sigmamyo_c * np.exp(-sigmamyo_d * (P_gh - sigmamyo_e)) ))

    # Psi_AT1RAA: effect of AT1-bound ANG II on afferent resistance
    residual[75] = Psi_AT1RAA - (0.8   + 0.2092 * (AT1R / AT1R_eq) - 0.0092 / (AT1R / AT1R_eq))
    
    # Psi_AT1REA: effect of AT1-bound ANG II on efferent resistance
    residual[76] = Psi_AT1REA - (0.925 + 0.0835 * (AT1R / AT1R_eq) - 0.0085 / (AT1R / AT1R_eq))

    # Psi_AT2RAA: effect of AT2R-bound Ang II on afferent resistance
    psiat2raa_a = 0.9
    psiat2raa_b = 1 - psiat2raa_a
    psiat2raa_c = 1

    if female:
        if m_AT2R:
            residual[77] = Psi_AT2RAA - 1
        else:
            residual[77] = Psi_AT2RAA - (Psi_AT2RAA_eq * (psiat2raa_a + psiat2raa_b * np.exp(-psiat2raa_c * (AT2R/AT2R_eq - 1))))
    else:
        residual[77] = Psi_AT2RAA - 1

    # Psi_AT2REA: effect of AT2R-bound Ang II on efferent resistance
    psiat2rea_a = 0.9
    psiat2rea_b = 1 - psiat2rea_a
    psiat2rea_c = 1
                
    if female:
        if m_AT2R:
            residual[78] = Psi_AT2REA - 1
        else:
            residual[78] = Psi_AT2REA - (Psi_AT2REA_eq * (psiat2rea_a + psiat2rea_b * np.exp(-psiat2rea_c * (AT2R/AT2R_eq - 1))))
    else:
        residual[78] = Psi_AT2REA - 1

    # Phi_twreab: tubular water reabsorption rate
    if not female:
        residual[79] = Phi_twreab - (A_twreab - 0.001* 1/(mu_adh) + (0.8 + 0.08 * np.tanh( 8.5 * (mu_Na-1)) ) * Phi_gfilt)
    else:
        residual[79] = Phi_twreab - ( 1.01*(A_twreab - 0.001* 1/(mu_adh)) + (0.8 + 0.08 * np.tanh( 8.5 * (mu_Na-1)) ) * Phi_gfilt )
    
    # mu_adh: effect of antidiuretic concentration on tubular water reabsorption rate
    residual[80] = mu_adh - (0.3313 + 0.8 / (1 + np.exp( 0.6 - 3.7 * np.log10(C_adh) )))

    # mu_Na
    residual[81] = mu_Na - ((Phi_ptsodreab + Phi_dtsodreab + Phi_cdsodreab) / (pt_sod_reab_EQ + dt_sod_reab_EQ + cd_sod_reab_EQ))

    # Phi_u: urine flow rate
    if (kappa_ACEi == 0) and (kappa_f == 0) and (kappa_f_md == 0) and (NSAID == 0):
        # softplus
        residual[82] = Phi_u - (0.0003 + np.log(1 + np.exp(Phi_gfilt - Phi_twreab - 0.0003)))
        #residual[82] = Phi_u - max(0.0003     , Phi_gfilt - Phi_twreab )
    else:  
        # softplus
        residual[82] = Phi_u - (0.0003*0.2 + np.log(1 + np.exp(Phi_gfilt - Phi_twreab - 0.0003*0.2)))
        #residual[82] = Phi_u - max(0.0003*0.2 , Phi_gfilt - Phi_twreab )

    # Phi_win: water intake
    if low_win_ind:
        # softplus
        residual[83] = Phi_win - np.log(1 + np.exp( 0.0177    / (3.9271  + 18.22*C_adh**-1.607) - 0.002))
        #residual[83] = Phi_win - max(0, 0.0177    / (3.9271  + 18.22*C_adh**-1.607) - 0.002)

    else:
        if female:
            # softplus
            residual[83] = Phi_win - np.log(1 + np.exp(0.9*(0.0078541 / (0.65451 + 18.22*np.log(1 + np.exp(C_adh))**-2.907) - 0.002)))
            #residual[83] = Phi_win - max(0, 0.9*(0.0078541 / (0.65451 + 18.22*max(0,C_adh)**-2.907) - 0.002))
        else:
            # softplus
            residual[83] = Phi_win - np.log(1 + np.exp(0.0078541 / (0.65451 + 18.22*C_adh**-2.907) - 0.002)) # the exponent of C_adh was changed from -1.607 to -2.907 since otherwise, all trajectories are constants
            #residual[83] = Phi_win - max(0, 0.0078541 / (0.65451 + 18.22*C_adh**-2.907) - 0.002) 
            
def run_simulation(female, rsna_scenario, hormone_scenario, hormone_influence: HormoneInfluence, days=1, save_results: bool = False):
    '''
    This function gathers initial values and model parameters based on gender and rsna scenario, 
    hormone configuration a during simulation runtime.
    All necessary information is formatted for the DAE solver and results are saved upon successfull solver termination.
    '''
    # time points
    tchange = 100
    num_days = days
    t0 = 0
    tf = tchange + num_days*1440 + hormone_influence.start_t

    # solve ODEs
    t_eval = np.linspace(t0, tf, num=int((tf/10)+1), dtype=int)
    print("t_eval:", t_eval)

    # species (15 species in ODE)
    # [V_ecf, vas, a_baro, N_adh, delta_ra, M_sod, N_al, AGT, PRC, AngI, AngII, AT1R, AT2R, Ang17, AngIV]
    initial_values = load_initial_values(get_initial_values_filename(female=female, rsna_scenario=rsna_scenario), column_name='initial_concentration', index_selection='all')
    model_parameters = load_file_contents_by_column(get_model_parameters_filename(female=female, scenario_context=rsna_scenario), column_name='parameter_value')
    
    # set hormone influence parameters
    hormone_influence_parameters = load_hormone_influence_on_parameters(hormone_scenario)
    hormone_influence.set_hormone_parameters(hormone_influence_parameters)

    initial_derivatives = np.zeros(len(initial_values))
    ode_indices = get_species_indices('ODE')
    differential_vars = [bool(i in ode_indices) for i in range(len(initial_values))]
    p = (female,) + (hormone_influence,) + tuple(model_parameters)

    dae_prob = de.DAEProblem(bp_model, initial_derivatives, initial_values, (t0,tf), p, differential_vars=differential_vars)
    try:
        sol = de.solve(dae_prob, de.IDA(), initializealg=de.ShampineCollocationInit(), saveat=t_eval) 
    except Exception as e:
        print(f"Simulation failed because of {e}")
        return

    if sol.errors is None:
        print("Simulation completed successfully.")
        if save_results:
            res = "minutes"
            save_simulation_results(sol, filename_results=get_simulation_result_filename(female, rsna_scenario, hormone_scenario, days), filename_parameters=get_simulation_parameter_values_filename(female, rsna_scenario, hormone_scenario, days), resolution=res)
    else:
        print(f"Simulation unsuccessful because of {sol.message}")

def run_hormone_influence_simulations(genders:list, rsna_scenarios:list, hormone_scenarios:list, hormone_influence: HormoneInfluence, days=60, save_results: bool=False):
    '''
    This function generates all possible combinations of genders, rsna scenarios and hormone scenarios
    and calls the run_simulation() function where initial values and model parameters are loaded according
    to the specific combination.
    '''
    # make sure to include the baseline scenario for baseline reference
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios

    for sex in genders:
        for rsna_scenario in rsna_scenarios:
            # in case of simulations for male, only run the baseline hormone configuration, do not include 
            # e2 or p4 influence on any parameters for now
            if not sex:
                sex_spedific_hormone_scenarios = ['baseline']
                print("For simulations for male only running baseline hormone configuration.")
            else:
                sex_spedific_hormone_scenarios = hormone_scenarios

            for hormone_scenario in sex_spedific_hormone_scenarios:
                print(f"Starting simulation of (female: {sex}, rsna scenario {rsna_scenario}, hormone scenario {hormone_scenario}) for {days} days")
                timer_start = time.time()
                if hormone_scenario == 'baseline':
                    run_simulation(sex, rsna_scenario, hormone_scenario, hormone_influence=hormone_influence, days=days, save_results=save_results)
                else:
                    run_simulation(sex, rsna_scenario, hormone_scenario, hormone_influence=hormone_influence, days=days, save_results=save_results)
                t_end = time.time()
                print(f"Simulation of (female: {sex}, rsna scenario {rsna_scenario}, hormone scenario {hormone_scenario}) for {days} days - wall time: {t_end - timer_start:.2f} seconds")

def calculate_relative_differences(baseline: pd.DataFrame, baseline_avg: pd.DataFrame, scenario_results: pd.DataFrame, species_under_consideration: list) -> pd.DataFrame:
    '''
    Function to calculate the species-specific relative difference between baseline and scenario simulations.
    (species scenario value - species baseline value)/species average value if species average values is greater than 10e-10,
    otherwise the difference is stated as 0 (in order to prevent division by zero).
    '''
    all_species = get_species_names_and_indices()
    if species_under_consideration:
        reduced_species = {s: all_species[s] for s in species_under_consideration if s in all_species}
    else:
        reduced_species = all_species
    
    # Downsample baseline and scenario results to the same time points 
    baseline_ds = downsample_df(baseline)
    scenario_results_ds = downsample_df(scenario_results)

    # Compute differences
    diffs = []
    for species_name, species_idx in reduced_species.items():
        species_avg = baseline_avg.at[species_idx, 'simulation_average']
        diff_series = [(sc[species_idx] - ba[species_idx])/species_avg if np.abs(species_avg) > 10e-10 else 0 for ba, sc in zip(baseline_ds["simulation_results"], scenario_results_ds["simulation_results"])]
        diffs.append(pd.Series(diff_series, name=species_name))

    # Assemble into one DataFrame with time
    df_diff = pd.concat([baseline_ds["t"]] + diffs, axis=1)
    return df_diff

def calculate_simulation_average(scenario_results: pd.DataFrame, hormone_influence: HormoneInfluence, species_under_consideration: list, cycle_phase: bool = False) -> pd.DataFrame:
    '''
    Function to calculate the species-specific average of the simulation result. Species can be selected via the list 'species_under_consideration'. 
    If 'cycle_phase' is True, the definition of cycle phases is used to calculate species-specific averages
    per cycle phase. The phase definition is currently restricted to a single file but can be changed to use
    a different definition file.
    '''
    all_species = get_species_names_and_indices()
    if species_under_consideration:
        reduced_species = {s : all_species[s] for s in species_under_consideration if s in all_species}
    else:
        reduced_species = all_species

    sim_data = np.vstack(scenario_results["simulation_results"])

    if cycle_phase:
        cycle_phase_definitions = load_file_contents(HUMAN_DATA_DIR_CSV.joinpath(get_cycle_phase_definition_filename()))
        cycle_phase_definitions["indices"] = cycle_phase_definitions.apply(lambda row: hormone_influence.get_relevant_time_indices_for_cycle_phase(len(scenario_results), row["start_day"], row["end_day"]), axis=1)
        
        cycle_phase_avgs = {}
        for _, row in cycle_phase_definitions.iterrows():
            idx = np.array(row['indices'])
            # Compute mean over selected timesteps (rows) for all species at once
            phase_means = sim_data[idx, :].mean(axis=0)
            cycle_phase_avgs[row['phase_name']] = {n: phase_means[i] for n, i in reduced_species.items()}

        avg_df = pd.DataFrame(cycle_phase_avgs).reset_index(names='species')
    else:
        phase_means = sim_data[hormone_influence.start_t:, :].mean(axis=0)
        avg_by_species = {n : phase_means[i] for n, i in reduced_species.items()}
        avg_df = pd.DataFrame.from_dict(avg_by_species, orient='index', columns=['simulation_average']).reset_index(names='species')
    
    gc.collect() # force garbage collection to free memory
    return avg_df

def evaluate_simulation_results(genders:list, rsna_scenarios:list, hormone_scenarios:list, hormone_influence: HormoneInfluence, days=60, save_results: bool = False):
    '''
    Evaluation of simulation results. For each combination of the values in the gender, rsna scenario and hormone scenario lists
    the simulation results are evaluated and saved to files. Evaluation of species averages over the whole
    simulation, menstrual cycle phase-dependent averages, and relative differences between baseline simulation and scenario-specific
    simulation results.
    '''
    # make sure to include the baseline scenario for baseline reference
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios

    for sex in genders:
        for rsna_scenario in rsna_scenarios:
            # first retrieve baseline data for comparison (hormone scenario 'baseline')
            baseline_file_name = get_simulation_result_filename(sex, rsna_scenario, 'baseline', days)
            baseline = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(baseline_file_name))

            for hormone_scenario in hormone_scenarios:
                gender = 'female' if sex else 'male'
                print(f"Evaluating simulation results for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
                result_file_name = get_simulation_result_filename(sex, rsna_scenario, hormone_scenario, days)
                scenario_results = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(result_file_name))
                # compare simulation results with baseline -> baseline data is the hormone scenario 'baseline' 

                # first make sure that the time points for the evaluation are the same
                if not np.array_equal(baseline['t'], scenario_results['t']):
                    print(f"Time points do not match for simulation {gender} {rsna_scenario} {hormone_scenario} {days} days. Reducing evaluation length to scenario length {len(scenario_results['t'])-1}.")
                    min_length = min(len(baseline['t']), len(scenario_results['t']))
                    baseline = baseline[: min_length-1]
                    scenario_results = scenario_results[: min_length-1]
                
                # evaluate average over the whole simulation time
                df_avg = calculate_simulation_average(scenario_results=scenario_results, hormone_influence=hormone_influence, species_under_consideration=[])
                print(f"P_ma average: {df_avg['simulation_average'][41]}")

                # evaluate average over the cycle phases
                df_avg_cycle_phase = calculate_simulation_average(scenario_results=scenario_results, hormone_influence=hormone_influence, species_under_consideration=[], cycle_phase=True)

                # evaluate relative differences to baseline
                df_diff = calculate_relative_differences(baseline, df_avg, scenario_results, species_under_consideration=[])
                
                if save_results:
                    #### store evaluation results
                    ## store averages
                    # total averages
                    avg_file_name = get_evaluation_average_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                    df_avg.to_csv(HUMAN_EVALUATION_DIR.joinpath(avg_file_name))
                    print(f"Saved evaluation results (averages) to {avg_file_name}")

                    # averages per cycle phase
                    avg_cycle_phase_file_name = get_evaluation_phase_average_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                    df_avg_cycle_phase.to_csv(HUMAN_EVALUATION_DIR.joinpath(avg_cycle_phase_file_name))
                    print(f"Saved evaluation results (averages per cycle phase) to {avg_cycle_phase_file_name}")

                    # store relative differences
                    evaluation_file_name = get_evaluation_difference_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                    df_diff.to_csv(HUMAN_EVALUATION_DIR.joinpath(evaluation_file_name))
                    print(f"Saved evaluation results (relative differences) to {evaluation_file_name}")

                gc.collect() # force garbage collection to free memory      

def show_simulation_results(genders: list, rsna_scenarios: list, hormone_scenarios: list, hormone_influence:HormoneInfluence, days: int = 30, species: list = []):
    ''''
    Generation of visualisations for all combinations of genders, rsna scenarios and hormone scenarios for simulation results
    and simulation evaluations.
    '''
    for sex in genders:
        gender = 'female' if sex else 'male'
        for rsna_scenario in rsna_scenarios:
            for hormone_scenario in hormone_scenarios:
                simulation_file_name = get_simulation_result_filename(sex, rsna_scenario, hormone_scenario, days)
                simulation_results = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(simulation_file_name))
                simulation_parameter_values = load_file_contents(HUMAN_SIMULATION_DIR.joinpath(get_simulation_parameter_values_filename(female, rsna_scenario, hormone_scenario, days)))
                vis = create_simulation_results_visualisation_hormone_influence(species, simulation_results, simulation_parameter_values, hormone_influence, gender=gender, rsna_scenario=rsna_scenario, hormone_scenario=hormone_scenario)
                vis.show()
        gender = 'female' if sex else 'male'
        for rsna_scenario in rsna_scenarios:
            for hormone_scenario in hormone_scenarios:
                evaluation_result_file_name = get_evaluation_difference_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                evaluation_results = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(evaluation_result_file_name), column_name='')
                if evaluation_results.empty:
                    print(f"No evaluation results found for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
                    continue   
                if not species:
                    print(f"No species selected for plotting evaluation results, skipping for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days.")
                    continue
                
                vis = create_evaluation_result_plots(species, evaluation_results, gender=gender, rsna_scenario=rsna_scenario, hormone_scenario=hormone_scenario)
                vis.show()

def show_simulation_averages(genders:list, rsna_scenarios:list, hormone_scenarios:list, days=60, species_under_consideration: list = []):
    # make sure to include the baseline scenario for baseline reference
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios
    
    # Visualisation of simulation averages in a plotly bar chart
    for sex in genders:
        for rsna_scenario in rsna_scenarios:
            scenario_avgs = {}
            for hormone_scenario in hormone_scenarios:
                gender = 'female' if sex else 'male'
                print(f"Evaluating average simulation results for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
                avgs_filename = get_evaluation_average_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                df_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(avgs_filename))
                scenario_avgs[hormone_scenario] = df_avg.at[41,'simulation_average']

            vis = create_simulation_average_plots(scenario_avgs, gender, rsna_scenario)
            vis.show()

def show_simulation_averages_per_cycle_phase(genders:list, rsna_scenarios:list, hormone_scenarios:list, days=60, species_under_consideration: list = []):
    # Visualisation of simulation averages in a plotly bar chart
    
    # make sure to include the baseline scenario for baseline reference
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios

    for sex in genders:
        for rsna_scenario in rsna_scenarios:
            scenario_avgs = {}
            for hormone_scenario in hormone_scenarios:
                gender = 'female' if sex else 'male'
                print(f"Evaluating average simulation results for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
                result_avgs_filename = get_evaluation_phase_average_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
                df_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(result_avgs_filename))
                scenario_avgs[hormone_scenario] = df_avg.to_dict(orient='index')[41] # index 41 => P_ma

            cycle_phase_data = pd.DataFrame(scenario_avgs).iloc[2:] # skip the species column and the index column

            vis = create_simulation_average_plots_per_cycle_phase(cycle_phase_data, gender, rsna_scenario)
            vis.show()

def show_MAP_comparison(genders:list, rsna_scenarios:list, hormone_scenarios:list, hormone_influence:HormoneInfluence, days=60):
    '''
    Visualisation of only the Mean Arterial Pressure trajectories for all simulations in the
    genders, rsna scenario and hormone scenario lists. 
    '''
    scenario_maps = pd.DataFrame()
    for sex in genders:
        gender = 'female' if sex else 'male'
        for rsna_scenario in rsna_scenarios:
            for hormone_scenario in hormone_scenarios:
                simulation_file_name = get_simulation_result_filename(sex, rsna_scenario, hormone_scenario, days)
                simulation_results = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(simulation_file_name))
                scenario_maps[hormone_scenario] = simulation_results["simulation_results"].str[41]

            scenario_maps['t'] = simulation_results['t']
            # get baseline results
            avgs_filename = get_evaluation_average_filename(sex, rsna_scenario, "baseline", days, file_type='csv')
            baseline_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(avgs_filename))
            baseline_avg_MAP = baseline_avg.at[41,'simulation_average']

            vis = create_MAP_comparison_visualisation_hormone_influence(scenario_maps, baseline_avg_MAP, hormone_influence, gender, rsna_scenario)
             
            vis.show()

if __name__ == "__main__":
    '''
    Below is an example of how to configure the blood pressure simulations, evaluations, visualisations
    of baseline scenarios, and hormone influence.
    '''
    timer_start = time.time()

    start_t = 1440  # start hormone influence after 1440 minutes (1 day) to account for transition period
            
    female = True
    days = 180 # ODE variables reach steady states after +- 1440 minutes so we skip the first day in the hormone influence simulations
    cycle_length = 28
    menstrual_cycles = int(days/cycle_length) # or int((days/cycle_length)/2) for hormone influence for half of the simulation duration; determine how many cycles should be simulated based on simulation length
    parameter_influence = "linear" # one of ["linear", "exp", "hill", "noise"]
    hormone_influence = HormoneInfluence(influence_type=parameter_influence, cycle_length=cycle_length, start_t=start_t, hormone_parameters={}, duration=menstrual_cycles)

    # hormone_influence.cycle.plot_hormone_profiles()
    # available hormone scenarios:
    # scenario 'baseline' - no hormone influence, will always be run to generate baseline data automatically for evaluation
    # hormone scenarios are listed in .model/blood_pressure/human_data/hormone_profile_*.csv
    hormone_scenarios = ['baseline', 'Literature']
    genders = [female]
    rsna_scenarios = ['rsna1']
    
    run_hormone_influence_simulations(genders=genders, rsna_scenarios=rsna_scenarios, hormone_scenarios=hormone_scenarios, days=days, hormone_influence=hormone_influence, save_results=True)
    show_simulation_results(genders=genders, rsna_scenarios=rsna_scenarios, hormone_scenarios=hormone_scenarios, days=days, hormone_influence=hormone_influence, species=[])
    evaluate_simulation_results(genders=genders, rsna_scenarios=rsna_scenarios, hormone_scenarios=hormone_scenarios, hormone_influence=hormone_influence, days=days, save_results=True)
    show_MAP_comparison(genders, rsna_scenarios, hormone_scenarios, hormone_influence, days)

    # select species to plot in a single plot
    # species_to_plot = get_ODE_species_names() + ['P_ma']
    species_to_plot = get_species_names_and_indices().keys()
   
    timer_end = time.time()
    print(f"Complete simulation, evaluation and visualisation of (genders: {genders}, rsna scenario {rsna_scenarios}, hormone scenario {hormone_scenarios}) for {days} days - wall time: {timer_end - timer_start:.2f} seconds")

    gc.collect() # force garbage collection to free memory
