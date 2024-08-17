"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel, Arad Gast
    Created: 01/10/21
    Edited: 31/05/24

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import os
import warnings
from src.training import *
from run_simulation import run_simulation
import argparse
from src.sensors_arrays import *

# Initialization
# warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

PIPE_CLEAN = 0
SANITY_CHECK_NUM_SOURCES = 1
SANITY_CHECK_ULA = 0
MRA_NUM_SENSORS = 4
scenario_dict = {
    "coherent": [],
    "non-coherent": [5],
}
num_virtual_sensors = SensorsArray(f"MRA-{MRA_NUM_SENSORS}" ).last_sensor_loc
if SANITY_CHECK_ULA:
    sensors_array_form = f'ULA-{num_virtual_sensors}'
else:
    sensors_array_form = f'MRA-{MRA_NUM_SENSORS}'
system_model_params = {
    "M": num_virtual_sensors - 1 if SANITY_CHECK_NUM_SOURCES else None,  # number of sources
    "N": num_virtual_sensors,                       # number of antennas
    "sensors_array_form": sensors_array_form,       # "ULA-7", "MRA-4" , "MRA-8"
    "missing_sensors_handle_method": Missing_senors_handle_method.zeros ,
    "T": 100,                                       # number of snapshots
    "snr": None,                                    # if defined, values in scenario_dict will be ignored
    "field_type": "Far",                            # Near, Far
    "signal_nature": None,                          # if defined, values in scenario_dict will be ignored
    "eta": 0,                                       # steering vector error
    "bias": 0,
    "sv_noise_var": 0,
    "doa_range": 75,                                # doa range in degrees
    "min_gap": 5                                    # minimal gap in degrees
}
model_config = {
    "model_type": "SubspaceNet",                # SubspaceNet, CascadedSubspaceNet, DeepCNN, TransMUSIC, DR_MUSIC
    "model_params": {}
}
if model_config.get("model_type") == "SubspaceNet":
    model_config["model_params"]["diff_method"] = "esprit"  # esprit, music_1D, music_2D
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["field_type"] = "Far"     # Near, Far

elif model_config.get("model_type") == "CascadedSubspaceNet":
    model_config["model_params"]["tau"] = 8

elif model_config.get("model_type") == "DeepCNN":
    model_config["model_params"]["grid_size"] = 361

training_params = {
    "samples_size": 1024 * 8 if PIPE_CLEAN else 1024 * 100,
    "train_test_ratio": .5,
    "training_objective": "angle",       # angle, range, source_estimation
    "batch_size": 256 if PIPE_CLEAN else 256,
    "epochs": 2 if PIPE_CLEAN else 100,
    "optimizer": "Adam",                        # Adam, SGD
    "learning_rate": 0.0001,
    "weight_decay": 1e-9,
    "step_size": 70,
    "gamma": 0.5,
    "true_doa_train": None,                 # if set, this doa will be set to all samples in the train dataset
    "true_range_train": None,                 # if set, this range will be set to all samples in the train dataset
    "true_doa_test": None,                  # if set, this doa will be set to all samples in the test dataset
    "true_range_test": None,                   # if set, this range will be set to all samples in the train dataset
    "criterion": "rmspe",                   # rmse, rmspe, mse, mspe, bce, cartesian
    "balance_factor": 1.0                # if None, the balance factor will be set to the default value -> 0.6
}
evaluation_params = {
    "criterion": "rmspe",                       # rmse, rmspe, mse, mspe, cartesian
    "balance_factor": training_params["balance_factor"],
    "models": {
                # "CascadedSubspaceNet": {"tau": 8},
                "SubspaceNet": {"tau": 8,
                                "diff_method": "esprit",
                                "field_type": "Far"},
                # "TransMUSIC": {},
            },
    "augmented_methods": [
        # "mvdr",
        # "r-music",
        # "esprit",
        # "music",
        # "music_2D",
    ],
    "subspace_methods": [
        "esprit",
        # "music_1d",
        # "root_music",
        # "mvdr",
        # "sps_root_music",
        # "sps_esprit",
        # "sps_music_1d"
        # "bb-music",
        # "music_2D",
        # "sps_music_2D",
        # "CRB"
    ]
}
simulation_commands = {
    "SAVE_TO_FILE": True,
    "CREATE_DATA": False,
    "LOAD_MODEL": True,
    "TRAIN_MODEL": True,
    "SAVE_MODEL": True,
    "EVALUATE_MODE": True,
    "PLOT_RESULTS": False
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('--snr', type=str, help='SNR value', default=None)
    parser.add_argument('--N', type=int, help='Number of antennas', default=None)
    parser.add_argument('--M', type=int, help='Number of sources', default=None)
    parser.add_argument('--S', type=int, help='Sensors Array', default=None)
    parser.add_argument('--field_type', type=str, help='Field type', default=None)
    parser.add_argument('--signal_nature', type=str, help='Signal nature', default=None)
    parser.add_argument('--model_type', type=str, help='Model type', default=None)
    parser.add_argument('--train', type=int, help='Train model', default=None)
    parser.add_argument('--train_criteria', type=str, help='Training criteria', default=None)
    parser.add_argument('--eval', type=int, help='Evaluate model', default=None)
    parser.add_argument('--eval_criteria', type=str, help='Evaluation criteria', default=None)
    parser.add_argument('--samples_size', type=int, help='Samples size', default=None)
    parser.add_argument('--train_test_ratio', type=int, help='Train test ratio', default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # torch.set_printoptions(precision=12)

    args = parse_arguments()
    if args.snr is not None:
        system_model_params["snr"] = int(args.snr)
    if args.N is not None:
        system_model_params["N"] = int(args.N)
    if args.M is not None:
        system_model_params["M"] = int(args.M)
    if args.S is not None:
        system_model_params["sensors_array_form"] = args.S
    # if args.S_M is not None:
    #     system_model_params["missing_sensors_handle_method"] = args.S_M   
    if args.field_type is not None:
        system_model_params["field_type"] = args.field_type
    if args.signal_nature is not None:
        system_model_params["signal_nature"] = args.signal_nature
    if args.model_type is not None:
        model_config["model_type"] = args.model_type
    if args.train is not None:
        simulation_commands["TRAIN_MODEL"] = args.train
    if args.train_criteria is not None:
        training_params["training_objective"] = args.train_criteria
    if args.eval is not None:
        simulation_commands["EVALUATE_MODE"] = args.eval
    if args.eval_criteria is not None:
        evaluation_params["criterion"] = args.eval_criteria
    if args.samples_size is not None:
        training_params["samples_size"] = args.samples_size
    if args.train_test_ratio is not None:
        training_params["train_test_ratio"] = args.train_test_ratio

    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)

