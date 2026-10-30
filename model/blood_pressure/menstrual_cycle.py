import numpy as np
import pandas as pd
from pathlib import Path
# the following lines ensure that we can import local modules with a path from the project root directory
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from scipy.interpolate import make_splrep, BSpline

# local imports
from model.blood_pressure.utils import minmax_scale_value_by_array
from model.blood_pressure.file_handling import load_file_contents_by_column, get_hormone_concentration_filename

class MenstrualCycle():
    def __init__(self, cycle_length=28, e2_first_peak_day=14, e2_second_peak_day=22, p4_peak_day=22, cycles=10):
        self.cycle_length = cycle_length
        self.cycles=cycles
        self.e2_first_peak_day = e2_first_peak_day
        self.e2_second_peak_day = e2_second_peak_day
        self.p4_peak_day = p4_peak_day

        t = np.arange(cycle_length * cycles)

        self.E2_profile = np.array(
            [self.estradiol_profile(t_i % cycle_length, cycle_length, cycles=10)
            for t_i in t]
        )
        self.P4_profile = np.array(
            [self.progesterone_profile(t_i % cycle_length, cycle_length, cycles=10)
            for t_i in t]
        )
        self.T_profile = np.array(
            [self.testosterone_profile(t_i % cycle_length, cycle_length, peak_day=13)
            for t_i in t]
        )
        
    def gaussian_1(self,t, center):
        f = 90*np.exp(-(((t-center)-self.e2_second_peak_day)**2)/15)+140*np.exp(-(((t-center)-self.e2_first_peak_day)**2)/2)  # pg/mL
        return f

    def gaussian_2(self,t, center):
        f = 50*np.exp(-(((t-center)-self.e2_first_peak_day)**2)/20) # pg/mL
        return f

    def gaussian_3(self,t, center):
        f  = ((52*np.exp((-((t-center)-self.p4_peak_day)**2)/10))+(3.18/2))*314.46/1000 # ng/ml
        return f

    def estradiol_profile(self, t, cycle_length=28, cycles=10):
        """Hormone profile for estradiol (E2) over a <cycle_length>-day cycle."""
        # Initial value
        e2 = 60.0   # pg/mL
        pds = cycles

        for i in range(-pds, pds + 1):
            e2 = e2 + self.gaussian_1(t, cycle_length * (i - 1)) + self.gaussian_2(t, cycle_length * (i - 1))
        
        return e2

    def get_scaled_estradiol_concentration(self, t):
        """Scaled hormone profile for estradiol (E2) over a <cycle_length>-day cycle."""
        e2 = self.estradiol_profile(t, self.cycle_length, self.cycles)
        # scale between 0 and 1
        # min 20: https://emedicine.medscape.com/article/2089003-overview?form=fpf
        e2_scaled = minmax_scale_value_by_array(e2, ref_min=20, ref_max=max(self.E2_profile), lower_bound=0, upper_bound=1)
        return e2_scaled

    def progesterone_profile(self, t, cycle_length=28, cycles=10):
        """Hormone profile for progesterone (P4) over a <cycle_length>-day cycle."""
        # Initial value
        p4 = 0.5    # ng/mL (average values)
        pds = cycles

        for i in range(-pds, pds + 1):
            p4 = p4 + self.gaussian_3(t, cycle_length * (i - 1)) - 0.5
        
        return p4

    def get_scaled_progesterone_concentration(self, t):
        """Scaled hormone profile for progesterone (P4) over a <cycle_length>-day cycle."""
        p4 = self.progesterone_profile(t, self.cycle_length, self.cycles)
        # scale between 0 and 1
        p4_scaled = minmax_scale_value_by_array(p4, ref_min=0.1, ref_max=max(self.P4_profile), lower_bound=0, upper_bound=1)
        return p4_scaled

    def testosterone_profile(self, t, cycle_length=28, peak_day=13):
        # The reference interval in women was 0.30–1.69 nmol/L (8.7–48.7 ng/dL) 
        # https://www.sciencedirect.com/science/article/pii/S0039128X12002863#s0080
        # we decided to use a cosine wave to approximate the monthly variation of testosterone in the female cycle as a first iteration
        # To convert testosterone concentrations from nmol/L to ng/mL, multiply by 0.3
        t_min = 0.30 * 0.3 # ng/mL
        t_max = 1.69 * 0.3 # ng/mL

        # frequency
        omega = 2 * np.pi / cycle_length

        # scaling
        amplitude = (t_max - t_min) / 2
        offset = (t_max + t_min) / 2

        # cosine wave
        y = offset + amplitude * np.cos(omega * (t - peak_day))

        return y

    def plot_hormone_profiles(self):
        plt.figure()
        plt.plot(self.E2_profile, label='E2 profile pg/mL')
        plt.plot(self.P4_profile, label='P4 profile ng/mL')
        #plt.plot(T_profile, label='T profile')
        plt.legend(loc='upper left')
        plt.title('Hormone Profiles Estrogen (E2), Progesterone (P4)')
        #plt.show()

    def scale_by_hormone_profile(self, t, profile):
        if profile.size == 0:
            return 0
        
        d = int(t/1440) % self.cycle_length
        hormone_value = profile[d]
        return hormone_value

class ContraceptiveCycle(MenstrualCycle):
    def __init__(self, cycle_length=28, cycles=10, profile_name:str="contraceptive"):
        super().__init__(cycle_length, cycles=cycles)
        # override the hormone profiles with values taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC11390121/
        # to simulate the effect of hormonal contraceptives

        self.E2_contraceptive_original:BSpline = self.load_contraceptive_hormone_profile(profile_name=profile_name, column_name="E2_pgmL")
        self.P4_contraceptive_original:BSpline = self.load_contraceptive_hormone_profile(profile_name=profile_name, column_name="P4_ngmL")
        self.contraceptive_metadata:pd.DataFrame = self.load_contraceptive_metadata(profile_name=profile_name + '_full')

        self.e2_first_peak_day_contraceptive = 19
        self.e2_second_peak_day_contraceptive = 6 #27

        t = np.arange(cycle_length * self.cycles)

        self.E2_profile_contraceptive = np.array(
            [self.estradiol_profile_contraceptive(t_i % cycle_length, cycle_length, cycles=10)
            for t_i in t]
        )
        self.P4_profile_contraceptive = np.array(
            [self.progesterone_profile_contraceptive(t_i % cycle_length)
            for t_i in t]
        )

    def load_contraceptive_metadata(self, profile_name:str='contraceptive_full'):
        metadata = load_file_contents_by_column(get_hormone_concentration_filename(profile_name), column_name='')

        # to align the cycles with influence of contraception and the naturally occuring cycles,
        # we perform a shift of the hormone profiles: the withdrawal period starts after day 21 of the cycle, 
        # lasts 7 days, and the first pill of the next cycle is taken on day 28+1. Thus, we shift the hormone profiles by 7 days 
        # to align the onset of menstruattion in the naturally occuring cycle with the withdrawal bleeding in the contraceptive cycle.

        shift_amount = -(22 - 1)  # Shift left by 21 positions
        reordered_df = pd.DataFrame(np.roll(metadata.values, shift_amount, axis=0), 
                                    columns=metadata.columns)

        new_labels = [str(i) for i in list(range(22, 29)) + list(range(1, 22))]
        reordered_df.index = new_labels
        return reordered_df

    def load_contraceptive_hormone_profile(self, profile_name:str="contraceptive", column_name:str="E2_pgmL"):
        # load the hormone profiles for contraceptive cycle from file
        # the file should have columns 'E2_pgmL', 'P4_ngmL' with hormone values for each day of the cycle
        time_profile = np.array(load_file_contents_by_column(get_hormone_concentration_filename(profile_name), column_name='day'))
        hormone_profile = np.array(load_file_contents_by_column(get_hormone_concentration_filename(profile_name), column_name=column_name))

        hormone_spline = make_splrep(time_profile, hormone_profile, bc_type='periodic')
        
        return hormone_spline 
    
    def get_scaled_estradiol_concentration(self, t):
        """Scaled hormone profile for estradiol (E2) over a <cycle_length>-day cycle under influence of contraceptive medication."""
        e2 = self.estradiol_profile_contraceptive(t, self.cycle_length, self.cycles)
        # scale between 0 and 1
        e2_scaled = minmax_scale_value_by_array(e2, ref_min=16, ref_max=max(self.E2_profile), lower_bound=0, upper_bound=1) # min 16: https://pmc.ncbi.nlm.nih.gov/articles/PMC11390121/
        return e2_scaled
    
    def estradiol_profile_contraceptive(self, t, cycle_length=28, cycles=10):
        """Hormone profile for estradiol (E2) over a <cycle_length>-day cycle under influence of contraceptive medication."""
        # Initial value
        e2 = 16.0   # pg/mL
        pds = cycles

        for i in range(-pds, pds + 1):
            e2 = e2 + self.gaussian_1_contraceptive(t, cycle_length * (i - 1)) + self.gaussian_2_contraceptive(t, cycle_length * (i - 1)) + self.gaussian_1_contraceptive(t, cycle_length * (i - 1))
        
        return e2
    
    def gaussian_1_contraceptive(self, t, center):
        peak_width = 1.75
        f = 23*np.exp(-(((t-center)-self.e2_second_peak_day_contraceptive)**2)/(2*peak_width**2)) # pg/mL
        return f

    def gaussian_2_contraceptive(self, t, center):
        peak_width = 0.7
        f = 26*np.exp(-(((t-center)-self.e2_first_peak_day_contraceptive)**2)/(2*peak_width**2)) # pg/mL
        return f
    
    def get_scaled_progesterone_concentration(self, t):
        """Scaled hormone profile for progesterone (P4) over a <cycle_length>-day cycle under influence of contraceptive medication."""
        p4 = self.progesterone_profile_contraceptive(t)
        # scale between 0 and 1
        p4_scaled = minmax_scale_value_by_array(p4, ref_min=0.25, ref_max=max(self.P4_profile), lower_bound=0, upper_bound=1)
        return p4_scaled
    
    def progesterone_profile_contraceptive(self, t):
        """Hormone profile for progesterone (P4) over a <cycle_length>-day cycle under influence of contraceptive medication."""
        # the average value of progesterone under contraception influence (based on https://pmc.ncbi.nlm.nih.gov/articles/PMC11390121/)
        # was 0.6 using the spline in function 'load_contraceptive_hormone_profile'
        return np.full_like(t, 0.6, dtype=float) # ng/mL

    def plot_hormone_profiles(self, metadata:bool=True):
        t = np.linspace(1, self.cycle_length*10, num=self.cycle_length*10)
        x = np.linspace(1, self.cycle_length, num=self.cycle_length)
        plt.figure()
        if not metadata:
            plt.plot(self.E2_contraceptive_original(t), label='E2 contraceptive profile pg/mL')
            plt.plot(self.P4_contraceptive_original(t), label='P4 contraceptive profile ng/mL')
        else:
            # estradiol
            E2_e = self.contraceptive_metadata['E2_stdev']
            E2_y = self.contraceptive_metadata['E2_pgmL']
            plt.errorbar(x, E2_y, yerr=E2_e, linestyle='None', marker='.', label='E2 standard deviation pg/mL')

            # progesterone
            P4_e = self.contraceptive_metadata['P4_stdev']
            P4_y = self.contraceptive_metadata['P4_ngmL']
            plt.errorbar(x, P4_y, yerr=P4_e, linestyle='None', marker='.', label='P4 standard deviation ng/mL')

        plt.plot(self.E2_profile_contraceptive, label='E2 profile pg/mL')
        plt.plot(self.P4_profile_contraceptive, label='P4 profile ng/mL')
        #plt.plot(T_profile, label='T profile')
        plt.legend(loc='upper right')
        plt.title('Hormone Profiles Estrogen (E2), Progesterone (P4) with oral contraception')

        plt.figure()
        plt.plot(self.get_scaled_estradiol_concentration(t), label='scaled E2 profile contraceptive pg/mL')
        plt.plot(self.get_scaled_progesterone_concentration(t), label='scaled P4 profile contraceptive ng/mL')
        plt.legend(loc='upper right')
        plt.title('Hormone Profiles Estrogen (E2) and Progesterone (P4) with oral contraception, scaled')
        plt.show()

    
if __name__ == "__main__":
    #cycle = MenstrualCycle(cycle_length=28, e2_first_peak_day=14, e2_second_peak_day=22, p4_peak_day=22)
    #cycle.plot_hormone_profiles()
    pill_cycle = ContraceptiveCycle(cycle_length=28)
    pill_cycle.plot_hormone_profiles()

