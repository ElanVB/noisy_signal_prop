
# imports
import numpy as np
import os, sys, pickle

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

# custom import
from theory import depth
from viz import get_colours
from numpy_simulation import *
from utils import load_experiment
from theory import critical_point

def perform_experiment(experiments):
    for i, experiment in enumerate(experiments):
            dist = experiment['dist']
            noise = experiment['noise']
            act = experiment['act']
            init = experiment['init']

            # run simulations for scenario
            noisy_signal_prop_simulations(dist, noise, act, init, seed=i)

def variance():
    experiments = [
        {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"underflow"},
        {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"overflow"},
        {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"underflow"},
        {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"overflow"},
        {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"}
    ]

    perform_experiment(experiments)


def correlation():
    # Compute experimental data
    experiments = [
        {"dist": "none", "noise": (None, None), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 2), "act":"relu", "init":"crit"}
    ]

    perform_experiment(experiments)

def fixed_point():
    # Compute experimental data
    experiments = [
        {"dist": "bern", "noise": ('prob_1', 0.1), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.2), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.3), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.4), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.5), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.7), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
        {"dist": "bern", "noise": ('prob_1', 0.9), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.1), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.4), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.55), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.7), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 0.85), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.0), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.15), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.3), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.45), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.6), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.75), "act":"relu", "init":"crit"},
        {"dist": "mult gauss", "noise": ('std', 1.9), "act":"relu", "init":"crit"}
    ]

    perform_experiment(experiments)

if __name__ == "__main__":
    # results directory
    results_dir = os.path.join(file_dir, "../results")

    # variance()
    # correlation()
    fixed_point()
