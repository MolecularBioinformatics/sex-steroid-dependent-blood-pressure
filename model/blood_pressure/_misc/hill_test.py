from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print("Python Path:", sys.path)

import os
print("Script Location:", os.path.abspath(__file__))
print("Current Working Directory:", os.getcwd())

import matplotlib.pyplot as plt

# local imports
from model.blood_pressure.visualisation import create_hill_plots
from model.blood_pressure.menstrual_cycle import MenstrualCycle

if __name__ == "__main__":
    hill_types = ["negative"]
    exps = [1, 2, 3, 4, 5]
    Ks = [0.2, 0.5, 0.7, 1.0]
    cycle = MenstrualCycle(cycle_length=28, e2_first_peak_day=14, e2_second_peak_day=22, p4_peak_day=22)
    ts = [(1 + cycle.get_scaled_estradiol_concentration(p) * 0.5)*cycle.get_scaled_progesterone_concentration(p) for p in range(1,57)]
    ts_p = [cycle.get_scaled_progesterone_concentration(p) for p in range(1,57)]

    for hill_type in hill_types:
        for K in Ks:
            fig = create_hill_plots(hill_type, exps, K, start=0, end=1, ts=ts)
            fig.show()

    plt.plot(ts, label="E2*P4")
    plt.plot(ts_p, label="P4")
    plt.xlabel("Time (days)")
    plt.ylabel("Hormone Concentration")
    plt.title("Hormone Profiles")
    plt.legend()
    plt.show()