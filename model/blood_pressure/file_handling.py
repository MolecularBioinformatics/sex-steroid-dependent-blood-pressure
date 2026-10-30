from pathlib import Path
import pandas as pd

# local imports

#### Data Directories start ####
DATA_DIR_PYTHON = Path('model/blood_pressure')
HUMAN_DATA_DIR_CSV = DATA_DIR_PYTHON.joinpath('human_data')

HUMAN_SIMULATION_DIR = DATA_DIR_PYTHON.joinpath('simulation_results')

HUMAN_EVALUATION_DIR = DATA_DIR_PYTHON.joinpath('evaluation_results')
#### Data Directories end ####

def load_initial_values(filename: str, column_name: str = 'initial_concentraion', index_selection:str = 'ODE') -> list:
    try:
        file_name = Path(filename)
        file_path = HUMAN_DATA_DIR_CSV.joinpath(file_name)
        file_data = list(pd.read_csv(file_path, sep=',')[column_name])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
        
    # select relevant indices depending on rsna scenario, preselection is the ODE scenario normotensive
    select_indices = get_species_indices(index_selection)

    file_data = [file_data[i] for i in select_indices]

    return file_data

def load_file_contents_by_column(filename: str, column_name: str = 'parameter_value'):
    file_name = Path(filename)
    file_path = HUMAN_DATA_DIR_CSV.joinpath(file_name)

    try:
        # if no file attribute is given, return the whole dataframe
        if not column_name:
            return pd.read_csv(file_path, sep=',')
        
        # if file attribute is given, return only the respective column as a list
        file_data = list(pd.read_csv(file_path, sep=',')[column_name])
        return file_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []

def load_file_contents(filename: str = 'model_parameters.csv', column_name='')-> pd.DataFrame:
    file_path = Path(filename)

    try:
        file_data = pd.read_csv(file_path, sep=',')
        # if no file attribute is given, return the whole dataframe
        if not column_name:
            return file_data
        
        # if file attribute is given, return only the respective column as a list
        return file_data[column_name]
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def load_hormone_influence_on_parameters(hormone_scenario: str = 'baseline')-> dict:
    file_name = HUMAN_DATA_DIR_CSV.joinpath(get_hormone_profile_filename(hormone_scenario=hormone_scenario))
    hormone_influence = load_file_contents(file_name, column_name='')
    # convert the two columns into a dictionary so that the parameters can be accessed by their name
    return dict(zip(hormone_influence["hormone_parameter_influence"], hormone_influence["parameter_value"]))

def save_hormone_influence_parameter_configuration(hormone_scenario: str = 'baseline', hormone_parameters: dict = {}):
    file_name = HUMAN_DATA_DIR_CSV.joinpath(get_hormone_profile_filename(hormone_scenario=hormone_scenario))
    hormone_influence_df = pd.DataFrame(list(hormone_parameters.items()), columns=['hormone_parameter_influence', 'parameter_value'])
    hormone_influence_df.to_csv(file_name, index=False)
    print(f"Saved hormone influence parameters to {file_name}")

def get_all_hormone_influence_parameter_configurations() -> list:   
    hormone_influence_files = HUMAN_DATA_DIR_CSV.glob('hormone_profile_*.csv')
    hormone_influence_file_names = [file.name.replace('hormone_profile_', '').replace('.csv', '') for file in hormone_influence_files]
    return hormone_influence_file_names

def get_all_evaluation_results_by_config_name(gender:str, rsna_scenario: str = 'rsna1', days: int = 30, evaluation_type:str='differences') -> list:
    evaluation_result_files = HUMAN_EVALUATION_DIR.glob(f'evaluation_results_{evaluation_type}_{gender}_{rsna_scenario}_*_{days}_days.csv')
    evaluation_result_file_names = [file.name.replace(f'evaluation_results_{evaluation_type}_{gender}_{rsna_scenario}_', '').replace(f'_{days}_days.csv', '') for file in evaluation_result_files]
    return evaluation_result_file_names

def save_simulation_results(sol, filename_results: str = 'simulation_results.pkl', filename_parameters: str = 'simulation_parameter_values.csv', resolution:str="days"):
    times = sol.t
    values = sol.u
    solution = pd.DataFrame({'t':times, 'simulation_results':values})

    # downsample the results according to resolution
    if resolution == "days":
        step = 144 # 1440 minutes in a day, and 10 minute resolution of t_eval, so 144 steps per day
    if resolution == "minutes":
        step = 1

    solution_downsampled = solution.iloc[::step]

    file_path_results = HUMAN_SIMULATION_DIR.joinpath(filename_results)
    solution_downsampled.to_pickle(file_path_results)
    print(f"Saved simulation results to {filename_results}")

    # retrieve parameter values affected by hormone influence
    parameter_values = sol.prob.p[1].parameter_values_over_time
    parameter_df = pd.DataFrame(parameter_values)
    parameter_df = parameter_df.drop_duplicates(subset=["t"], keep="last").sort_values(by="t")
    parameter_df_downsampled = parameter_df.iloc[::step]
    file_path_parameters = HUMAN_SIMULATION_DIR.joinpath(filename_parameters)
    parameter_df_downsampled.to_csv(file_path_parameters, index=False)
    print(f"Saved simulation parameter values to {filename_parameters}")

def load_df_from_file(file_path: str = 'simulation_results.pkl') -> pd.DataFrame:
    simulation = pd.read_pickle(file_path)
    return simulation

def get_hormone_concentration_filename(hormone_scenario: str = 'contraceptive') -> str:
    return f'hormone_concentrations_{hormone_scenario}.csv'

def get_cycle_phase_definition_filename() -> str:
    return 'cycle_phase_definitions.csv'

def get_evaluation_difference_filename(female: bool, rsna_scenario: str = 'rsna1', hormone_scenario: str = 'baseline', days: int = 30, file_type: str = 'png') -> str:
    sex = 'female' if female else 'male'
    return f'evaluation_results_differences_{sex}_{rsna_scenario}_{hormone_scenario}_{days}_days.{file_type}'

def get_evaluation_average_filename(female: bool, rsna_scenario: str = 'rsna1', hormone_scenario: str = 'baseline', days: int = 30, file_type: str = 'png') -> str:
    sex = 'female' if female else 'male'
    return f'evaluation_results_averages_{sex}_{rsna_scenario}_{hormone_scenario}_{days}_days.{file_type}'

def get_evaluation_phase_average_filename(female: bool, rsna_scenario: str = 'rsna1', hormone_scenario: str = 'baseline', days: int = 30, file_type: str = 'png') -> str:
    sex = 'female' if female else 'male'
    return f'evaluation_results_averages_per_cycle_phase_{sex}_{rsna_scenario}_{hormone_scenario}_{days}_days.{file_type}'

def get_simulation_result_filename(female: bool, rsna_scenario: str = 'rsna1', hormone_scenario: str = 'baseline', days: int = 30) -> str:
    sex = 'female' if female else 'male'
    return f'simulation_results_{sex}_{rsna_scenario}_{hormone_scenario}_{days}_days.pkl'

def get_simulation_parameter_values_filename(female: bool, rsna_scenario: str = 'rsna1', hormone_scenario: str = 'baseline', days: int = 30) -> str:
    sex = 'female' if female else 'male'
    return f'simulation_parameter_values_{sex}_{rsna_scenario}_{hormone_scenario}_{days}_days.csv'

def get_noise_filename(female: bool, rsna_scenario: str = 'rsna1', days: int = 30, noise_name:str='perlin_0_5') -> str:
    sex = 'female' if female else 'male'
    return f'noise_{sex}_{rsna_scenario}_{noise_name}_{days}_days.csv'

def get_hormone_profile_filename(hormone_scenario: str = 'baseline') -> str:
    return f'hormone_profile_{hormone_scenario}.csv'

def get_model_parameters_filename(species:str = 'human', female: bool=True, scenario_context: str='rsna1') -> str:
    sex = 'female' if female else 'male'
    return f'{species}_{sex}_model_parameters_{scenario_context}.csv'

def get_initial_values_filename(species:str = 'human', female: bool=True, rsna_scenario: str='rsna1') -> str:
    sex = 'female' if female else 'male'
    return f'{species}_{sex}_steady_states_{rsna_scenario}.csv'

def get_species_indices(index_selection:str = 'ODE'):
    if index_selection == 'ODE':
        return [28, 34, 45, 47, 49, 50, 55, 61, 64, 66, 67, 68, 69, 70, 71]
    elif index_selection == 'algebraic':
        possible_indices = list(range(84))
        return [i for i in possible_indices if i not in get_species_indices('ODE')]
    else: 
        return list(range(84))
    
def get_ODE_species_names():
    return ['V_ecf', 'vas', 'a_baro', 'N_adh', 'delta_ra', 'M_sod', 'N_al', 'AGT', 'PRC', 'AngI', 'AngII', 'AT1R', 'AT2R', 'Ang17', 'AngIV']

def get_species_names_and_indices() -> dict:
    # select any file of the human csv steady state files to retrieve the species names and their order
    file_name = HUMAN_DATA_DIR_CSV.joinpath("human_female_steady_states_rsna1.csv")
    # Read CSV, header row automatically ignored
    df = pd.read_csv(file_name)

    # Suppose the column with names is called "name"
    name_to_index = {name: idx for idx, name in enumerate(df["species"])}

    return name_to_index