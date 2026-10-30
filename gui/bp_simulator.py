import streamlit as st
from pathlib import Path
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
#print("Python Path:", sys.path)

import os
#print("Script Location:", os.path.abspath(__file__))
#print("Current Working Directory:", os.getcwd())

# local imports
from model.blood_pressure.main_model import HUMAN_SIMULATION_DIR, evaluate_simulation_results, get_simulation_result_filename, load_df_from_file, run_hormone_influence_simulations
from model.blood_pressure.hormone_influence import HormoneInfluence, HormoneInfluenceContraceptive
from model.blood_pressure.visualisation import create_MAP_comparison_visualisation, create_MAP_comparison_visualisation_hormone_influence, create_evaluation_result_plots, create_simulation_average_plots, create_simulation_average_plots_line, create_simulation_average_plots_per_cycle_phase, create_simulation_results_visualisation_hormone_influence
from model.blood_pressure.file_handling import HUMAN_EVALUATION_DIR, get_evaluation_average_filename, get_evaluation_difference_filename, get_evaluation_phase_average_filename, get_simulation_parameter_values_filename, load_file_contents, load_hormone_influence_on_parameters, save_hormone_influence_parameter_configuration, get_all_evaluation_results_by_config_name, get_all_hormone_influence_parameter_configurations

### functions for Streamlit app ###
def run_simulation_and_evaluation(days, female, rsna_scenario, hormone_scenarios, hormone_influence):
    st.markdown("### 🔬 Simulating blood pressure")

    progress = st.progress(0)

    run_hormone_influence_simulations(genders=[female], rsna_scenarios=[rsna_scenario], hormone_scenarios=hormone_scenarios, days=days, hormone_influence=hormone_influence, save_results=True)
    progress.progress(50)
    st.markdown("### 🔬 Evaluating blood pressure simulation results")
    evaluate_simulation_results(genders=[female], rsna_scenarios=[rsna_scenario], hormone_scenarios=hormone_scenarios, hormone_influence=hormone_influence, days=days, save_results=True)
    progress.progress(100)
    
def get_simulation_figure(days, female, rsna_scenario, hormone_scenario, hormone_influence):
    gender = 'female' if female else 'male'
    simulation_file_name = get_simulation_result_filename(female, rsna_scenario, hormone_scenario, days)
    simulation_results = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(simulation_file_name))
    simulation_parameter_values = load_file_contents(HUMAN_SIMULATION_DIR.joinpath(get_simulation_parameter_values_filename(female, rsna_scenario, hormone_scenario, days)))
    simulation_figure = create_simulation_results_visualisation_hormone_influence([], simulation_results, simulation_parameter_values, hormone_influence, gender=gender, rsna_scenario=rsna_scenario, hormone_scenario=hormone_scenario)
    return simulation_figure

def get_MAP_comparison_figure(days, female, rsna_scenario, hormone_scenarios, hormone_influence):
    scenario_maps = pd.DataFrame()
    gender = 'female' if female else 'male'

    for hormone_scenario in hormone_scenarios:
        simulation_file_name = get_simulation_result_filename(female, rsna_scenario, hormone_scenario, days)
        simulation_results = load_df_from_file(HUMAN_SIMULATION_DIR.joinpath(simulation_file_name))
        scenario_maps[hormone_scenario] = simulation_results["simulation_results"].str[41]

    scenario_maps['t'] = simulation_results['t']
    # get baseline results
    avgs_filename = get_evaluation_average_filename(sex, rsna_scenario, "baseline", days, file_type='csv')
    baseline_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(avgs_filename))
    baseline_avg_MAP = baseline_avg.at[41,'simulation_average']

    # MAP_comparison_figure = create_MAP_comparison_visualisation(scenario_maps, baseline_avg_MAP, gender, rsna_scenario)
    MAP_comparison_figure = create_MAP_comparison_visualisation_hormone_influence(scenario_maps, baseline_avg_MAP, hormone_influence, gender, rsna_scenario)
    
    return MAP_comparison_figure

def show_evaluation_results(days, female, rsna_scenario, hormone_scenario):
    gender = 'female' if female else 'male'
    evaluation_result_file_name = get_evaluation_difference_filename(gender, rsna_scenario, hormone_scenario, days, file_type='csv')
    evaluation_results = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(evaluation_result_file_name), column_name='')
    evaluation_figure = create_evaluation_result_plots([], evaluation_results, gender=gender, rsna_scenario=rsna_scenario, hormone_scenario=hormone_scenario)
        
    st.write(f"Evaluation of {gender} simulation with hormone influences for scenario {hormone_scenario}, {rsna_scenario}.")
    st.plotly_chart(evaluation_figure, config = {'scrollZoom': True}, width='stretch')

def show_evaluation_results_avg(days, female, rsna_scenario, hormone_scenarios, line_plot=False):
    # make sure to include the null scenario for baseline information
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios
    
    # Visualisation of simulation averages in a plotly bar chart
    gender = 'female' if female else 'male'
    
    scenario_avgs = {}
    for hormone_scenario in hormone_scenarios:
        gender = 'female' if sex else 'male'
        #print(f"Evaluating average simulation results for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
        avgs_filename = get_evaluation_average_filename(sex, rsna_scenario, hormone_scenario, days, file_type='csv')
        df_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(avgs_filename))
        scenario_avgs[hormone_scenario] = df_avg.at[41,'simulation_average']

    if line_plot:
        evaluation_avg_figure = create_simulation_average_plots_line(scenario_avgs, gender, rsna_scenario)
    else:
        evaluation_avg_figure = create_simulation_average_plots(scenario_avgs, gender, rsna_scenario)
    st.plotly_chart(evaluation_avg_figure, config = {'scrollZoom': True}, width='stretch')

def show_evaluation_results_cycle_phase(days, female, rsna_scenario, hormone_scenarios):
    # make sure to include the null scenario for baseline information
    if 'baseline' not in hormone_scenarios:
        hormone_scenarios = ['baseline'] + hormone_scenarios

    gender = 'female' if female else 'male'
    scenario_avgs = {}
    for hormone_scenario in hormone_scenarios:
        #print(f"Evaluating average simulation results for {gender}, {rsna_scenario}, {hormone_scenario}, {days} days")
        result_avgs_filename = get_evaluation_phase_average_filename(gender, rsna_scenario, hormone_scenario, days, file_type='csv')
        df_avg = load_file_contents(HUMAN_EVALUATION_DIR.joinpath(result_avgs_filename))
        scenario_avgs[hormone_scenario] = df_avg.to_dict(orient='index')[41] # index 41 => P_ma

    cycle_phase_data = pd.DataFrame(scenario_avgs).iloc[2:] # skip the species column and the index column

    evaluation_cycle_phase_figure = create_simulation_average_plots_per_cycle_phase(cycle_phase_data, gender, rsna_scenario)
    st.plotly_chart(evaluation_cycle_phase_figure, config = {'scrollZoom': True}, width='stretch')

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Sex-Specific Long-Term Blood Pressure Simulation")
st.sidebar.markdown("## Configuration")

hormone_parameters = {}

# Initialize session state for hormone configuration
if "new_hormone_config" not in st.session_state:
    st.session_state.new_hormone_config = {
        "name": "null",
        "parameters": {
            "e2_ACE": 1.0,
            "e2_ACE2": 1.0,
            "e2_NRS": 1.0,
            "e2_AT1R": 1.0,
            "e2_AT2R": 1.0,
            "e2_ALD": 1.0,
            "p4_AT1R": 1.0,
            "p4_ALD": 1.0,
            "p4_MR": 1.0,
            "noise_effect_strength": 0.01
        },
        "editing": False
    }

if st.sidebar.button("New Hormone Configuration"):
    st.session_state.new_hormone_config["editing"] = True

if st.session_state.new_hormone_config["editing"]:
    # Text input for the configuration name
    st.session_state.new_hormone_config["name"] = st.sidebar.text_input(
        "New Hormone Configuration Name",
        value=st.session_state.new_hormone_config["name"],
        placeholder="e.g., combined_0_5",
        key="new_hormone_scenario_name"
    )

    # Sliders for hormone parameters
    st.session_state.new_hormone_config["parameters"]["e2_ACE"] = st.sidebar.slider(
        'Estradiol on ACE', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_ACE"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["e2_ACE2"] = st.sidebar.slider(
        'Estradiol on ACE2', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_ACE2"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["e2_NRS"] = st.sidebar.slider(
        'Estradiol on NRS', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_NRS"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["e2_AT1R"] = st.sidebar.slider(
        'Estradiol on AT1R', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_AT1R"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["e2_AT2R"] = st.sidebar.slider(
        'Estradiol on AT2R', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_AT2R"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["e2_ALD"] = st.sidebar.slider(
        'Estradiol on ALD', -10.0, 10.0, st.session_state.new_hormone_config["parameters"]["e2_ALD"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["p4_AT1R"] = st.sidebar.slider(
        'Progesterone on AT1R', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["p4_AT1R"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["p4_ALD"] = st.sidebar.slider(
        'Progesterone on ALD', 0.0, 10.0, st.session_state.new_hormone_config["parameters"]["p4_ALD"], 0.25
    )
    st.session_state.new_hormone_config["parameters"]["p4_MR"] = st.sidebar.slider(
        'Competitive inhibition of MR activity by Progesterone', 0.0, 1.0, st.session_state.new_hormone_config["parameters"]["p4_MR"], 0.1
    )
    st.session_state.new_hormone_config["parameters"]["noise_effect_strength"] = st.sidebar.slider(
        'Noise Effect Strength', 0.0, 0.1, st.session_state.new_hormone_config["parameters"]["noise_effect_strength"], 0.005
    )

    # Save configuration button
    if st.sidebar.button("Save Configuration"):
        if st.session_state.new_hormone_config["name"]:
            save_hormone_influence_parameter_configuration(
                st.session_state.new_hormone_config["name"],
                st.session_state.new_hormone_config["parameters"]
            )
            hormone_scenario = st.session_state.new_hormone_config["name"]
            st.sidebar.success(f"Hormone configuration '{st.session_state.new_hormone_config['name']}' saved successfully!")
            st.session_state.new_hormone_config["editing"] = False  # Exit editing mode
        else:
            st.sidebar.error("Please provide a name for the hormone configuration.")

if "show_hormone_config" not in st.session_state:
    st.session_state.show_hormone_config = False

if st.sidebar.button("Toggle Hormone Configuration"):
    st.session_state.show_hormone_config = not st.session_state.show_hormone_config

if st.session_state.show_hormone_config:
    all_hormone_scenarios = get_all_hormone_influence_parameter_configurations()
    simulation_hormone_scenario = st.sidebar.selectbox(
        f"Show Hormone Influence Parameter Configuration",
        all_hormone_scenarios
    )
    hormone_influence_parameters = load_hormone_influence_on_parameters(simulation_hormone_scenario)
    st.sidebar.write(f"Hormone Influence Parameters for scenario '{simulation_hormone_scenario}':")
    for param, value in hormone_influence_parameters.items():
        st.sidebar.write(f"- {param}: {value}")

rsna_scenario = 'rsna1'

col1, col2 = st.columns(2)

# Left column: Simulation
with col1:
    st.header("Simulation")
    # Check if simulation_figure exists in session state
    if "simulation_figure" not in st.session_state:
        st.session_state.simulation_figure = None
    
    simulation_length = st.slider('Time Range (days)', 30, 22000, 180, 10)  # min, max, default, step

    genders = ["female", "male"]
    sex = st.selectbox('Sex', genders)
    female = bool(sex == "female")
    
    if female:
        hormone_influence_start_latest = simulation_length
        hormone_influence_start_default = 1 # start earliest after 10 days
        hormone_influence_start = st.slider('Hormone Influence Start (days)', 1, hormone_influence_start_latest, hormone_influence_start_default)  # min, max, default

        cycle_length = st.slider('Cycle Length (days)', 21, 40, 28)  # min, max, default
        max_duration = int((simulation_length - hormone_influence_start)/cycle_length)
        hormone_influence_duration_cycles = st.slider('Hormone Influence Duration (cycles)', 0, max_duration, max_duration)  # min, max, default
        hormone_influence_start_t = (1+hormone_influence_start) * 1440  # start hormone influence after at least 10 days, given in minutes (after 10 days the system has reached steady state)
        hormone_influence_type = st.selectbox(
                label="Type of Hormone Influence on Parameters", 
                options=["linear", "exp", "hill", "noise"]
            )
        
        contraception = st.selectbox('Simulating Contraception', ['no', 'yes'])

        if contraception == 'no':
            e2_first_peak_day = st.text_input(
                "Day of First Estradiol Peak (E2)",
                value=14,
                placeholder="e.g., 14",
                key="e2_first_peak_day"
            )
            e2_second_peak_day = st.text_input(
                "Day of Second Estradiol Peak (E2)",
                value=22,
                placeholder="e.g., 22",
                key="e2_second_peak_day"
            )
            p4_peak_day = st.text_input(
                "Day of Progesterone Peak (P4)",
                value=22,
                placeholder="e.g., 22",
                key="p4_peak_day"
            )

            e2_on_p4_effect = st.slider('E2 on P4 Effect Strength', 0.0, 2.0, 0.2, 0.1)

            hormone_influence = HormoneInfluence(influence_type=hormone_influence_type, cycle_length=cycle_length, start_t=hormone_influence_start_t, hormone_parameters=hormone_parameters, duration=hormone_influence_duration_cycles, e2_first_peak_day=int(e2_first_peak_day), e2_second_peak_day=int(e2_second_peak_day), p4_peak_day=int(p4_peak_day), e2_on_p4=e2_on_p4_effect)
        else:
            hormone_influence = HormoneInfluenceContraceptive(influence_type=hormone_influence_type, cycle_length=cycle_length, start_t=hormone_influence_start_t, hormone_parameters=hormone_parameters, duration=hormone_influence_duration_cycles)
    else:
        hormone_influence = HormoneInfluence(cycle_length=1, start_t=14400, hormone_parameters={}, duration=0)

    all_hormone_scenarios = get_all_hormone_influence_parameter_configurations()
    simulation_hormone_scenarios = st.multiselect(
        f"Run Simulation with Hormone Configuration",
        all_hormone_scenarios,
        default=['baseline']
    )
    if st.button("Run Simulation"):
        run_simulation_and_evaluation(simulation_length, female, rsna_scenario, simulation_hormone_scenarios, hormone_influence)
    if st.button("Show Simulation Results"):
        st.session_state.simulation_figure = get_simulation_figure(simulation_length, female, rsna_scenario, simulation_hormone_scenarios[0], hormone_influence)
    if st.button("Show MAP Comparison"):
        st.session_state.simulation_figure = get_MAP_comparison_figure(simulation_length, female, rsna_scenario, simulation_hormone_scenarios, hormone_influence)
    if st.session_state.simulation_figure:
        st.plotly_chart(st.session_state.simulation_figure, config={'scrollZoom': True}, width='stretch')

# Right column: Evaluation
with col2:
    st.header("Evaluation")
    show_eval = st.checkbox("Show Relative Difference to Baseline")
    show_eval_avg = st.checkbox("Average Difference to Baseline")
    show_eval_cycle_phase = st.checkbox("Average Difference to Baseline Per Cycle Phase")

    if show_eval:
        all_hormone_scenarios = get_all_evaluation_results_by_config_name(sex, rsna_scenario, simulation_length, evaluation_type='averages')
        hormone_scenario_for_comparison = st.selectbox(
            f"Available Hormone Scenarios for Evaluation",
            all_hormone_scenarios
        )
        show_evaluation_results(simulation_length, female, rsna_scenario, hormone_scenario_for_comparison)
    if show_eval_avg:
        line_plot = st.checkbox("Line Plot")
        all_hormone_scenarios = get_all_evaluation_results_by_config_name(sex, rsna_scenario, simulation_length, evaluation_type='averages')
        hormone_scenarios_for_comparison = st.multiselect(
            f"Available Hormone Scenarios for Evaluation",
            all_hormone_scenarios,
            default=['baseline']
        )
        show_evaluation_results_avg(simulation_length, female, rsna_scenario, hormone_scenarios_for_comparison, line_plot=line_plot)
    if show_eval_cycle_phase:
        all_hormone_scenarios = get_all_evaluation_results_by_config_name(sex, rsna_scenario, simulation_length, evaluation_type='averages_per_cycle_phase')
        hormone_scenarios_for_comparison = st.multiselect(
            f"Available Hormone Scenarios for Evaluation",
            all_hormone_scenarios,
            default=['baseline']
        )
        show_evaluation_results_cycle_phase(simulation_length, female, rsna_scenario, hormone_scenarios_for_comparison)
