import numpy as np
import noise

# local imports
from model.blood_pressure.menstrual_cycle import MenstrualCycle, ContraceptiveCycle
from model.blood_pressure.utils import hill_pos, hill_neg

class HormoneInfluence:
    def __init__(self, name="", influence_type="linear", cycle_length=28, e2_first_peak_day=14, e2_second_peak_day=22, p4_peak_day=22, start_t=14400, hormone_parameters:dict={}, duration=1, e2_on_p4 = 0.2):
        self.name = name
        self.influence_type = influence_type
        self.cycle = MenstrualCycle(cycle_length=cycle_length, e2_first_peak_day=e2_first_peak_day, e2_second_peak_day=e2_second_peak_day, p4_peak_day=p4_peak_day, cycles=max(duration, 10))
        self.start_t = start_t
        self.hormone_parameters = hormone_parameters
        self.duration = duration  # duration in cycles
        self.parameter_selection = ["c_ACE", "c_ACE2", "N_rs", "c_AT1R", "c_AT2R", "N_als_eq", "gamma_MR"]
        self.parameter_values_over_time = {p: [] for p in self.parameter_selection}
        self.parameter_values_over_time["t"] = []
        self.e2_on_p4 = e2_on_p4  # fraction of E2 effect on P4 profile

    def set_hormone_parameters(self, hormone_parameters:dict):
        self.hormone_parameters = hormone_parameters

    def apply_hormone_influence_on_parameters(self, model_parameters:list, t:int):
        method = self.influence_type

        if method not in ['linear', 'exp', 'hill', 'noise']:
            print(f"Hormone influence method '{method}' not recognized. No hormone influence applied.")
            return model_parameters
        
        self.parameter_values_over_time['t'].append(t)

        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters

        if t <= self.start_t or self.duration == 0 or (self.duration > 0 and t > (self.cycle.cycle_length*1440*(self.duration+1)) + self.start_t) or self.hormone_parameters == {}:
            self.parameter_values_over_time['c_ACE'].append(c_ACE)
            self.parameter_values_over_time['c_ACE2'].append(c_ACE2)
            self.parameter_values_over_time['N_rs'].append(N_rs)
            self.parameter_values_over_time['c_AT1R'].append(c_AT1R)
            self.parameter_values_over_time['c_AT2R'].append(c_AT2R)
            self.parameter_values_over_time['N_als_eq'].append(N_als_eq)
            self.parameter_values_over_time['gamma_MR'].append(gamma_MR)

            if self.hormone_parameters == {}:
                print("No hormone parameters defined for hormone influence application.")

            return model_parameters

        # implement fading of hormone influence within the first cycle after hormone influence duration to avoid non-termination due to discontinuity
        elif t >= self.start_t and self.duration != 0 and (self.duration > 0 and t >= (self.cycle.cycle_length*1440*(self.duration)) + self.start_t and t < (self.cycle.cycle_length*1440*(self.duration+1)) + self.start_t) and not self.hormone_parameters == {}:
            t_in_cycle = t - self.cycle.cycle_length*1440*self.duration - self.start_t
            hormone_potency = 1.0 - (t_in_cycle / (1440*self.cycle.cycle_length))
            model_parameters_result = self.apply_hormone_influence_on_parameters_linear(model_parameters, t, hormone_potency)
        else:
            if method == 'linear':
                model_parameters_result = self.apply_hormone_influence_on_parameters_linear(model_parameters, t)
            elif method == 'exp':
                model_parameters_result = self.apply_hormone_influence_on_parameters_exp(model_parameters, t)
            elif method == 'hill':
                model_parameters_result = self.apply_hormone_influence_on_parameters_hill(model_parameters, t)
            else:  # method == 'noise'
                model_parameters_result = self.apply_hormone_influence_on_parameters_noise(model_parameters, t)

        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters_result
        
        self.parameter_values_over_time['c_ACE'].append(c_ACE)
        self.parameter_values_over_time['c_ACE2'].append(c_ACE2)
        self.parameter_values_over_time['N_rs'].append(N_rs)
        self.parameter_values_over_time['c_AT1R'].append(c_AT1R)
        self.parameter_values_over_time['c_AT2R'].append(c_AT2R)
        self.parameter_values_over_time['N_als_eq'].append(N_als_eq)
        self.parameter_values_over_time['gamma_MR'].append(gamma_MR)

        return model_parameters_result

    def apply_hormone_influence_on_parameters_linear(self, model_parameters:list, t:int, hormone_potency:float=1.0):

        # with this:
        #
        # c_AT1R = c_AT1R * (1 + e2_AT1R_effect * e2_t
        #                   + p4_AT1R_effect * p4_t)
        # 
        # we get linear, unbounded hormone effects, additive accumulation when multiple parameters are hormone-regulated, 
        # strong coupling between hormone amplitude and parameter magnitude, 
        # and this explains why the more parameters are affected, the lower we have to tune effect strengths to avoid non-termination
        # this is fragile in long-term DAE models, since small changes in hormone amplitude or parameter effect strength can lead to large changes in parameter values over time

        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters
        
        ### currently selected parameters that are influenced by hormones estradiol (E2), progesterone (P4), and testosterone (T) ###

        # calculate e2, p4, testosterone values for the time point t (divide by 1440 since the hormone profiles are in days)
        # scale values by the range of values in the specific hormone curve
        # in order to start cycle influence at cycle day 1, we need to substract the start_t offset from t before calculating the hormone concentration!
        # otherwise, the hormone influence would start at a random cycle day depending on the start_t value, shifting the evaluation results!
        cycle_t = t - self.start_t
        e2_t = self.cycle.get_scaled_estradiol_concentration(t=cycle_t/1440.) * hormone_potency
        p4_t = self.cycle.get_scaled_progesterone_concentration(t=cycle_t/1440.) * hormone_potency
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        #t_t = minmax_scale_value_by_array(testosterone_profile(t=t/1440., cycle_length=cycle_length), T_profile, lower_bound=0, upper_bound=1)

        # "The presence of estrogen enhances the effectiveness of P4" and existence of e2 and p4 receptors in renal tissue
        # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007848 and
        # https://karger.com/nef/article-pdf/149/2/77/4289696/000541352.pdf
        
        p4_t_adj = (1 + self.e2_on_p4 * e2_t) * p4_t
        #print(f"E2: {e2_t}, P4: {p4_t}, Adjusted P4: {p4_t_adj}")

        # ACE activity decreases with E2
        e2_ACE_effect = -self.hormone_parameters.get('e2_ACE', 0) #https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext#google_vignette
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ACE_effect = self.hormone_parameters['t_ACE'] # https://pubmed.ncbi.nlm.nih.gov/20608988/
        c_ACE = c_ACE * (1 + e2_ACE_effect * e2_t)
        
        # ACE2 activity increases with E2
        # https://www.researchgate.net/publication/285961341_Protective_regulation_of_the_ACE2ACE_gene_expression_by_oestrogen_in_human_atrial_tissue: 
        # the treatment with E2 resulted in the significant downregulation of ACE-mRNA (75±8% vs. 100% control, p<0.05) and an increase in the ACE2-mRNA (137±16%; p<0.05)

        # ACE2 activity decreases with E2: https://www.nature.com/articles/s41440-025-02400-y
        e2_ACE2_effect = self.hormone_parameters.get('e2_ACE2',0) 
        c_ACE2 = c_ACE2 * (1 + e2_ACE2_effect * e2_t)  

        # N_rs showed downregulation by E2
        # https://www.ahajournals.org/doi/10.1161/01.cir.95.1.39?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub
        e2_NRS_effect = -self.hormone_parameters.get('e2_NRS', 0) 
        N_rs = N_rs * (1 + e2_NRS_effect * e2_t)

        # AT1R activity decreases with E2, 
        # increases with P4,
        # increases with T
        e2_AT1R_effect = -self.hormone_parameters.get('e2_AT1R', 0) # https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext, https://pmc.ncbi.nlm.nih.gov/articles/PMC1828744/, https://pubmed.ncbi.nlm.nih.gov/9631868/
        p4_AT1R_effect = self.hormone_parameters.get('p4_AT1R', 0) # https://academic.oup.com/jcem/article-abstract/81/3/998/2649580?redirectedFrom=PDF&login=false, https://www.ahajournals.org/doi/10.1161/01.cir.102.15.1828
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_AT1R_effect = self.hormone_parameters['t_AT1R'] # https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2023.1198090/full#B145, https://journals.physiology.org/doi/full/10.1152/ajpheart.01277.2007
        c_AT1R = c_AT1R * (1 + e2_AT1R_effect * e2_t + p4_AT1R_effect * p4_t_adj) 

        # AT2R activity increases with E2,
        e2_AT2R_effect = self.hormone_parameters.get('e2_AT2R', 0) # human: https://journals.sagepub.com/doi/abs/10.1177/1535370217718808, mice: https://journals.physiology.org/doi/full/10.1152/ajprenal.00145.2002
        c_AT2R = c_AT2R * (1 + e2_AT2R_effect * e2_t)  

        # Aldosterone equilibrium concentration decreases with E2,
        # increases with P4,
        # decreases with T
        # contrary study results, e2 effect on aldosterone is included with sign in parameters file to test for both directions
        e2_ALD_effect = self.hormone_parameters.get('e2_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        p4_ALD_effect = -self.hormone_parameters.get('p4_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ALD_effect = -self.hormone_parameters['t_ALD'] # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        N_als_eq  = max(N_als_eq * (1 + p4_ALD_effect * p4_t_adj + e2_ALD_effect * e2_t), 0)
    
        # competitive binding of mineralocorticoid receptor (MR) activity by progesterone (P4)
        p4_MR_effect = self.hormone_parameters.get('p4_MR', 0) #https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        gamma_MR = (1 - p4_MR_effect) + p4_MR_effect * hill_neg(x=p4_t_adj, S=0.2, n=1) # 0.2 is the P4 concentration at which we have 50% of the maximal competitive inhibition effect, n is the hill exponent determining the steepness of the curve, taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/

        return [N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR]

    # version 2 of hormonal influence using exp for bounded effects
    def apply_hormone_influence_on_parameters_exp(self, model_parameters:list, t:int):
        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters
        
        ### currently selected parameters that are influenced by hormones estradiol (E2), progesterone (P4), and testosterone (T) ###

        # calculate e2, p4, testosterone values for the time point t (divide by 1440 since the hormone profiles are in days)
        # scale values by the range of values in the specific hormone curve
        # in order to start cycle influence at cycle day 1, we need to substract the start_t offset from t before calculating the hormone concentration!
        # otherwise, the hormone influence would start at a random cycle day depending on the start_t value, shifting the evaluation results!
        cycle_t = t - self.start_t
        e2_t = self.cycle.get_scaled_estradiol_concentration(t=cycle_t/1440.)
        p4_t = self.cycle.get_scaled_progesterone_concentration(t=cycle_t/1440.)
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        #t_t = minmax_scale_value_by_array(testosterone_profile(t=t/1440., cycle_length=cycle_length), T_profile, lower_bound=0, upper_bound=1)

        # "The presence of estrogen enhances the effectiveness of P4" and existence of e2 and p4 receptors in renal tissue
        # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007848 and
        # https://karger.com/nef/article-pdf/149/2/77/4289696/000541352.pdf
        
        p4_t_adj = (1 + self.e2_on_p4 * e2_t) * p4_t

        # ACE activity decreases with E2
        e2_ACE_effect = -self.hormone_parameters.get('e2_ACE', 0) #https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext#google_vignette
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ACE_effect = self.hormone_parameters['t_ACE'] # https://pubmed.ncbi.nlm.nih.gov/20608988/
        c_ACE = c_ACE * np.exp(e2_ACE_effect * e2_t) # -> concept of ec50 , since h_e2 almost similar to e2_t with threshold close to 1 and small hill exponent
        
        # ACE2 activity increases with E2
        # https://www.researchgate.net/publication/285961341_Protective_regulation_of_the_ACE2ACE_gene_expression_by_oestrogen_in_human_atrial_tissue: 
        # the treatment with E2 resulted in the significant downregulation of ACE-mRNA (75±8% vs. 100% control, p<0.05) and an increase in the ACE2-mRNA (137±16%; p<0.05)

        # ACE2 activity decreases with E2: https://www.nature.com/articles/s41440-025-02400-y
        e2_ACE2_effect = self.hormone_parameters.get('e2_ACE2', 0) 
        c_ACE2 = c_ACE2 * np.exp(e2_ACE2_effect * e2_t)  

        # N_rs showed downregulation by E2
        # https://www.ahajournals.org/doi/10.1161/01.cir.95.1.39?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub
        e2_NRS_effect = -self.hormone_parameters.get('e2_NRS', 0) 
        N_rs = N_rs * np.exp(e2_NRS_effect * e2_t)

        # AT1R activity decreases with E2, 
        # increases with P4,
        # increases with T
        #
        e2_AT1R_effect = -self.hormone_parameters.get('e2_AT1R', 0) # https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext, https://pmc.ncbi.nlm.nih.gov/articles/PMC1828744/
        p4_AT1R_effect = self.hormone_parameters.get('p4_AT1R', 0) # https://academic.oup.com/jcem/article-abstract/81/3/998/2649580?redirectedFrom=PDF&login=false, https://www.ahajournals.org/doi/10.1161/01.cir.102.15.1828
    
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_AT1R_effect = self.hormone_parameters['t_AT1R'] # https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2023.1198090/full#B145, https://journals.physiology.org/doi/full/10.1152/ajpheart.01277.2007
        c_AT1R = c_AT1R * np.exp(e2_AT1R_effect * e2_t + p4_AT1R_effect * p4_t_adj) 

        # AT2R activity increases with E2,
        e2_AT2R_effect = self.hormone_parameters.get('e2_AT2R', 0) # human: https://journals.sagepub.com/doi/abs/10.1177/1535370217718808, mice: https://journals.physiology.org/doi/full/10.1152/ajprenal.00145.2002
        c_AT2R = c_AT2R * np.exp(e2_AT2R_effect * e2_t)  

        # Aldosterone equilibrium concentration decreases with E2,
        # increases with P4,
        # decreases with T
        # because of contrary study results, e2 effect on aldosterone is included with sign in parameters file to test for both directions)
        e2_ALD_effect = self.hormone_parameters.get('e2_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        p4_ALD_effect = -self.hormone_parameters.get('p4_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ALD_effect = -self.hormone_parameters['t_ALD'] # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        N_als_eq  = N_als_eq * np.exp(p4_ALD_effect * p4_t_adj + e2_ALD_effect * e2_t)  
        
        # competitive inhibition of of mineralocorticoid receptor (MR) activity by progesterone (P4)
        p4_MR_effect = self.hormone_parameters.get('p4_MR', 0) #https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        gamma_MR = (1 - p4_MR_effect) + p4_MR_effect * hill_neg(x=p4_t_adj, S=0.2, n=1) # 0.2 is the P4 concentration at which we have 50% of the maximal competitive inhibition effect, n is the hill exponent determining the steepness of the curve, taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/

        return [N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR]

    # version 3 of hormonal influence using hill kinetics and exp for bounded effects
    def apply_hormone_influence_on_parameters_hill(self, model_parameters:list, t:int):
        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters
        
        ### currently selected parameters that are influenced by hormones estradiol (E2), progesterone (P4), and testosterone (T) ###

        # calculate e2, p4, testosterone values for the time point t (divide by 1440 since the hormone profiles are in days)
        # scale values by the range of values in the specific hormone curve
        # in order to start cycle influence at cycle day 1, we need to substract the start_t offset from t before calculating the hormone concentration!
        # otherwise, the hormone influence would start at a random cycle day depending on the start_t value, shifting the evaluation results!
        cycle_t = t - self.start_t
        e2_t = self.cycle.get_scaled_estradiol_concentration(t=cycle_t/1440.)
        p4_t = self.cycle.get_scaled_progesterone_concentration(t=cycle_t/1440.)
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        #t_t = minmax_scale_value_by_array(testosterone_profile(t=t/1440., cycle_length=cycle_length), T_profile, lower_bound=0, upper_bound=1)

        # "The presence of estrogen enhances the effectiveness of P4" and existence of e2 and p4 receptors in renal tissue
        # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007848 and
        # https://karger.com/nef/article-pdf/149/2/77/4289696/000541352.pdf
        
        p4_t_adj = (1 + self.e2_on_p4 * e2_t) * p4_t

        e2_t_hill = hill_pos(e2_t, n=3, S=0.5)
        p4_t_hill = hill_pos(p4_t_adj, n=3, S=0.2)

    
        # ACE activity decreases with E2
        e2_ACE_effect = -self.hormone_parameters.get('e2_ACE', 0) #https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext#google_vignette
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ACE_effect = self.hormone_parameters['t_ACE'] # https://pubmed.ncbi.nlm.nih.gov/20608988/
        c_ACE = c_ACE * np.exp(e2_ACE_effect * e2_t_hill) # -> concept of ec50 , since h_e2 almost similar to e2_t with threshold close to 1 and small hill exponent
        
        # ACE2 activity increases with E2
        # https://www.researchgate.net/publication/285961341_Protective_regulation_of_the_ACE2ACE_gene_expression_by_oestrogen_in_human_atrial_tissue: 
        # the treatment with E2 resulted in the significant downregulation of ACE-mRNA (75±8% vs. 100% control, p<0.05) and an increase in the ACE2-mRNA (137±16%; p<0.05)

        # ACE2 activity decreases with E2: https://www.nature.com/articles/s41440-025-02400-y
        e2_ACE2_effect = self.hormone_parameters.get('e2_ACE2', 0) 
        c_ACE2 = c_ACE2 * np.exp(e2_ACE2_effect * e2_t_hill)  

        # N_rs showed downregulation by E2
        # https://www.ahajournals.org/doi/10.1161/01.cir.95.1.39?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub
        e2_NRS_effect = -self.hormone_parameters.get('e2_NRS', 0) 
        N_rs = N_rs * np.exp(e2_NRS_effect * e2_t_hill)

        # AT1R activity decreases with E2, 
        # increases with P4,
        # increases with T
        #
        e2_AT1R_effect = -self.hormone_parameters.get('e2_AT1R', 0) # https://academic.oup.com/cardiovascres/article-abstract/53/3/672/327796?redirectedFrom=fulltext, https://pmc.ncbi.nlm.nih.gov/articles/PMC1828744/
        p4_AT1R_effect = self.hormone_parameters.get('p4_AT1R', 0) # https://academic.oup.com/jcem/article-abstract/81/3/998/2649580?redirectedFrom=PDF&login=false, https://www.ahajournals.org/doi/10.1161/01.cir.102.15.1828
    
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_AT1R_effect = self.hormone_parameters['t_AT1R'] # https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2023.1198090/full#B145, https://journals.physiology.org/doi/full/10.1152/ajpheart.01277.2007
        c_AT1R = c_AT1R * np.exp(e2_AT1R_effect * e2_t_hill + p4_AT1R_effect * p4_t_hill) 

        # AT2R activity increases with E2,
        e2_AT2R_effect = self.hormone_parameters.get('e2_AT2R', 0) # human: https://journals.sagepub.com/doi/abs/10.1177/1535370217718808, mice: https://journals.physiology.org/doi/full/10.1152/ajprenal.00145.2002
        c_AT2R = c_AT2R * np.exp(e2_AT2R_effect * e2_t_hill)  

        # Aldosterone secreation rate decreases with E2,
        # increases with P4,
        # decreases with T
        # because of contrary study results, e2 effect on aldosterone is included with sign in parameters file to test for both directions)
        e2_ALD_effect = self.hormone_parameters.get('e2_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        p4_ALD_effect = -self.hormone_parameters.get('p4_ALD', 0) # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        # comment out testosterone effect for now, since we focus on estradiol and progesterone first
        # t_ALD_effect = -self.hormone_parameters['t_ALD'] # https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        N_als_eq  = N_als_eq * np.exp(p4_ALD_effect * p4_t_hill + e2_ALD_effect * e2_t_hill)  
        
        # competitive inhibition of of mineralocorticoid receptor (MR) activity by progesterone (P4)
        p4_MR_effect = self.hormone_parameters.get('p4_MR', 0) #https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/
        gamma_MR = (1 - p4_MR_effect) + p4_MR_effect * hill_neg(x=p4_t_adj, S=0.2, n=1) # 0.2 is the P4 concentration at which we have 50% of the maximal competitive inhibition effect, n is the hill exponent determining the steepness of the curve, taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11155591/

        return [N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR]

    def apply_hormone_influence_on_parameters_noise(self, model_parameters:list, t:int):
        effect_strength = self.hormone_parameters.get('noise_effect_strength', 0.01)  # default effect strength if not specified
        if effect_strength == 0:
            return model_parameters
        
        N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR = model_parameters

        # apply random influence via perlin noise
        random_effect = np.exp(self.get_noise(t) * effect_strength)  # scale to influence strength
        c_ACE = c_ACE * random_effect
        c_ACE2 = c_ACE2 * random_effect
        N_rs = N_rs * random_effect
        c_AT1R = c_AT1R * random_effect
        c_AT2R = c_AT2R * random_effect
        N_als_eq  = N_als_eq * random_effect
        gamma_MR = random_effect
        
        return [N_rsna, R_aass, R_eass, P_B, P_go, C_gcf, eta_ptsodreab_eq, eta_dtsodreab_eq, eta_cdsodreab_eq, K_vd, K_bar, R_bv, N_adhs_eq, T_adh, Phi_sodin_const, N_als_eq, C_K, T_al, N_rs, X_PRCPRA, h_renin, h_AGT, h_AngI, h_AngII, h_Ang17, h_AngIV, h_AT1R, h_AT2R, k_AGT, c_ACE, c_Chym, c_NEP, c_ACE2, c_IIIV, c_AT1R, c_AT2R, AT1R_eq, AT2R_eq, Psi_AT2RAA_eq, Psi_AT2REA_eq, ALD_eq, pt_sod_reab_EQ, dt_sod_reab_EQ, cd_sod_reab_EQ, A_twreab, gamma_MR]

    def get_noise(self, t: float) -> float:
        val = noise.pnoise1(t)
        return val
    
    def get_relevant_time_indices_for_cycle_phase(self, simulation_length: int, cycle_phase_start: int, cycle_phase_end: int) -> list:
        # Start positions of each cycle
        # divide start_t by 10 to convert minutes to indices (10 min per index)
        bases = np.arange(int(self.start_t/10), simulation_length, self.cycle.cycle_length*144)
        
        # Offsets within each cycle
        offsets = np.arange((cycle_phase_start-1)*144, cycle_phase_end*144 + 1)
        
        # Combine bases and offsets using broadcasting
        indices = (bases[:, None] + offsets).ravel()
        
        # Filter out indices beyond n
        indices = indices[indices < simulation_length]
        
        return indices
    

class HormoneInfluenceContraceptive(HormoneInfluence):
    '''
    Implementation of a menstrual cycle under the influence of oral contraception, using the ContraceptiveCycle class.
    Since this class inherits from HormoneInfluence, all methods are available in this class as well.
    '''
    def __init__(self, name="", influence_type="linear", cycle_length=28, start_t=1440, hormone_parameters:dict={}, duration=1):
        self.name = name
        self.influence_type = influence_type
        self.cycle = ContraceptiveCycle(cycle_length=cycle_length, cycles=max(duration, 10))
        self.start_t = start_t
        self.hormone_parameters = hormone_parameters
        self.duration = duration  # duration in cycles
        self.parameter_selection = ["c_ACE", "c_ACE2", "N_rs", "c_AT1R", "c_AT2R", "N_als_eq", "gamma_MR"]
        self.parameter_values_over_time = {p: [] for p in self.parameter_selection}
        self.parameter_values_over_time["t"] = []
        self.e2_on_p4 = 0 # fraction of E2 effect on P4 profile
