# Female-specific long-term blood pressure regulation
This study incorporates both E2 and P4 impacts on blood pressure regulation with sufficient detail to distinguish the effects of distinct menstrual cycle phases. While the model successfully captures key trends and aligns with qualitative observations from previous literature, it represents a simplified physiological system. By decoupling the confounded influences and dynamics of hormonal fluctuations, this extended framework provides a robust foundation for sex-specific, personalized blood pressure analysis and predictive clinical tools.

## Python Environment
Create and activate the python environment as follows:
```
(conda clean --all -y) # clearing conda cache, optional
conda env create -f environment.yml
conda activate sex_specific_bp (env name is configured in environment.yml)
```
The environment.yml was created using
```
conda env export --no-builds > environment.yml 
```
## Julia
You do NOT have to install julia manually. This is solved via the environment.yml and julia will be installed inside the conda environment.

## Run blood pressure simulations
When running the model for the first time it might take some time for julia to download and install the necessary packages.
### GUI
The GUI implementation is located in ./gui/bp_simulator.py. To start the interactive GUI, open a terminal and navigate to the root folder ./. Executing the command
```
streamlit run gui/bp_simulator.py
```
will open a new tab in your browser at http://localhost:8501/. Settings are located in the top right corner of the page.
When running this command for the first time it might take some time for julia to download and install the necessary packages.
Check the console for Streamlit logs. When executing the command for the first time, Streamlit might print the following to the console:
```
Welcome to Streamlit!

      If you'd like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.

      Email: 
```
Where you can enter your email or opt out. The GUI will open in your browser once you entered the email adress or just pressed enter.

In the left side bar, hormone configurations can be created, read and updated. 
In the center under 'Simulation', new simulations can be configured with respect to menstrual cycle settings and hormonal influence, general settings such as simulation duration or the selection of a hormone configuration. Only existing hormone configurations can be selected. Previous simulation results generated with the same hormone configuration and simulation duration will be overwritten. For simulations without hormone influence, select 'baseline' configuration with default hormone effect strength of 0.
Start a simulation by clicking the button 'Run Simulation', visualise results via 'Show Simulation Results'. When selecting multiple hormone configurations and then clicking on 'Show MAP Comparison', all MAP trajectories of the respective configurations will be shown in a single plot.
Under 'Evaluation' are three possibilities for visualisation of evaluations. Only one can be shown at a time, and only successfull simulation runs will be selectable from the drop-down menues. Note that availability of results also depends on the simulation duration under 'Simulation': When simulation duration slider is at 180 days, evaluation options will only be listed in the dropdown for this exact length. If a specific simulation is missing from the dropdown, try generating it again or check the console output for errors.

The visualisations generated with plotly are interactive. Single-click on a species in the legend disables the trajectory while double-clicking a species in the legend disables all other species BUT this one. Can be used to visualise multiple species at once in more details. Further actions such as panning, zoom settings, autoscaling or reset are found in the top right corner of each plot.

### Local execution
Navigate to ./model/blood_pressure/main_model.py. The file contains a main which demonstrates the use of the functionality.
Run the command
```
<pyhton / path to your python executable> main_model.py
```
where you either use the keyword 'python' or insert the path pointing to the file '<...>/python.exe' on you machine.
When running this command for the first time it might take some time for julia to download and install the necessary packages.


### Hormone Configurations
Hormone configurations are .csv files stored in the folder ./model/blood_pressure/human_data. Naming of the files is 'hormone_profile_\<configuration name\>.csv'.
The configurations should contain the following fields:
```
hormone_parameter_influence,parameter_value
e2_ACE,0.25
e2_ACE2,0.37
e2_NRS,0.38
e2_AT1R,0.3
e2_AT2R,0.6
e2_ALD,-0.5
p4_AT1R,1
p4_ALD,0.6
p4_MR,0.8
```
In case a parameter is not listed in the configuration file, a value of 0.0 is assumed during simulation runtime.

### Menstrual Cycle Definition
#### Menstrual Cylce Phases
Menstrual cycle phases are defined to facilitate the analysis of menstrual cycle-based model species differences. Currently, this is implemented via the definitions in the file ./model/blood_pressure/human_data/cycle_phase_definitions.csv. If the default menstrual cycle length of 28 days is changed, the menstrual cycle phase definitions have to be adapted accordingly. Currently, this is only possible via changing the values manually in the file and not possible via the GUI.
```
phase_name,start_day,end_day
Follicular phase (FP), 1, 14
Luteal phase (LP), 15, 28
Early follicular phase (EFP), 3, 5
Late follicular phase (LFP), 11, 14
Mid-luteal phase (MLP), 19, 22
```

### Simulation Results
Simulation results are stored in the folder ./model/blood_pressure/simulation_results. For each simulation, two files are added to the folder:
```
simulation_parameter_values_<gender>_<rsna scenario>_<hormone configuration name>_<simulation duration>.csv
simulation_results_<gender>_<rsna scenario>_<hormone configuration name>_<simulation duration>.pkl
```
Evaluation files for averages per model species, averages per cycle phase per model species and difference per time step per model species are written to the folder ./model/blood_pressure/evaluation_results in the following form:
```
evaluation_results_averages_<gender>_<rsna scenario>_<hormone configuration name>_<simulation duration>.csv
evaluation_results_averages_per_cycle_phase_<gender>_<rsna scenario>_<hormone configuration name>_<simulation duration>.csv
evaluation_results_differences_<gender>_<rsna scenario>_<hormone configuration name>_<simulation duration>.csv
```

## Computational Model
### Baseline Model
The baseline computational model is written in MATLAB code and can be obtained from https://github.com/Layton-Lab/bp-K. 

Baseline steady state values and parameter values are located in the files
```
./model/blood_pressure/human_data/human_female_steady_states_rsna1.csv
./model/blood_pressure/human_data/human_female_steady_states_rsna2_5.csv
./model/blood_pressure/human_data/human_male_steady_states_rsna1.csv
./model/blood_pressure/human_data/human_male_steady_states_rsna2_5.csv

./model/blood_pressure/human_data/human_female_model_parameters_rsna1.csv
./model/blood_pressure/human_data/human_female_model_parameters_rsna2_5.csv
./model/blood_pressure/human_data/human_female_model_parameters_rsna1.csv
./model/blood_pressure/human_data/human_female_model_parameters_rsna2_5.csv
```

Note that analysis has mainly been done with rsna scenario '1' (normotension).

### Extension of the Baseline Model

#### Softplus
```
Cal with max() function:
residual[54] = C_al - max( 1, N_al * ALD_eq )
Cal with softplus approximation:
residual[54] = C_al - (1 + np.log(1 + np.exp(N_al * ALD_eq - 1)))
```
```
Nadhs with max() function:
residual[48] = N_adhs - (N_adhs_eq * (max( 0, C_sod - Nadhs_a) + max( 0, epsilon_aum - 1 ) - delta_ra) / 3)
Nadhs with softplus approximation:
N_adhs_tmp = N_adhs_eq * (np.log(1 + np.exp(C_sod - Nadhs_a)) + np.log( 1 + np.exp(epsilon_aum - 1 )))
residual[48] = N_adhs - np.log(1 + np.exp((N_adhs_tmp - (N_adhs_eq * delta_ra)) / 3))
```
```
Φsodin with max() function:
residual[27] = Phi_sodin - max(0,sodin_A/(sodin_B+sodin_C*C_al**sodin_D)+sodin_E)
Φsodin with softplus approximation:
residual[27] = Phi_sodin - np.log(1 + np.exp(sodin_A/(sodin_B+sodin_C*C_al**sodin_D)+sodin_E))
```
```
Φusod with max() function:
residual[26] = Phi_usod - max(0,( Phi_dtsod - Phi_cdsodreab ))
Φusod with softplus approximation:
residual[26] = Phi_usod - np.log(1 + np.exp( Phi_dtsod - Phi_cdsodreab ))
```
```
Φwin with max() function:
residual[83] = Phi_win - max(0, 0.0177 / (3.9271 + 18.22*C_adh**-1.607) - 0.002)
Φwin with softplus approximation:
residual[83] = Phi_win - np.log(1 + np.exp( 0.0177 / (3.9271 + 18.22*C_adh**-1.607) - 0.002))
```
```
Φu with max() function:
residual[82] = Phi_u - max(0.0003*0.2 , Phi_gfilt - Phi_twreab )
Φu with softplus approximation:
residual[82] = Phi_u - (0.0003*0.2 + np.log(1 + np.exp(Phi_gfilt - Phi_twreab -
0.0003*0.2)))
```

#### Hormonal Influence
Implementation can be found in ./model/blood_pressure/hormone_influence.py.

### Graph-based overview
A graph-based overview of the model was created with https://www.yworks.com/yed-live/. The folder ./graphs contains a .graphml file which can be uploaded in the web interface.
```
BloodPressureModel_Hormones.graphml
```
