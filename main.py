"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

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
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
from run_simulation import run_simulation

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

if __name__ == "__main__":
    # torch.set_printoptions(precision=12)
    # hold values for different scenarios, currently only for SNR and signal nature
    scenario_dict = {
        "coherent": [],
        "non-coherent": []
    }

    system_model_params = {
        "N": 5,
        "M": 2,
        "T": 100,
        "snr": 10,                         # if defined, values in scenario_dict will be ignored
        "field_type": "Near",
        "signal_nature": "non-coherent",  # if defined, values in scenario_dict will be ignored
        "eta": 0,
        "bias": 0,
        "sv_noise_var": 0
    }
    model_config = {
        "model_type": "SubspaceNet",
        "diff_method": "esprit",
        "tau": 3,
        "field_type": "Far"
    }
    training_params = {
        "samples_size": 1024 * 10,
        "train_test_ratio": 0.1,
        "training_objective": "range",
        "batch_size": 32,
        "epochs": 2,
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "weight_decay": 1e-9,
        "step_size": 70,
        "gamma": 0.5
    }
    evaluation_params = {
        "criterion": "rmspe",
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
            # "r-music",
            # "mvdr",
            # "sps-r-music",
            # "sps-esprit",
            "sps-music_1d"
            # "bb-music",
            # "music_2D"
        ]
    }
    simulation_commands = {
        "SAVE_TO_FILE": False,
        "CREATE_DATA": True,
        "LOAD_MODEL": False,
        "TRAIN_MODEL": False,
        "SAVE_MODEL": False,
        "EVALUATE_MODE": True,
        "PLOT_RESULTS": False
    }

    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)

