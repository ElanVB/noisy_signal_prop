# imports
import os
import numpy as np

# global variables
file_dir = os.path.dirname(os.path.realpath(__file__))

def load_experiment(test, data_names, relative_results_dir):
    data = {}
    for dist in test["distributions"]:
        data[dist["dist"]] = {}

        for act in test["activations"]:
            data[dist["dist"]][act] = {}

            for init in test["inits"]:
                data[dist["dist"]][act][init] = {}

                for name in data_names:
                    base_path = os.path.join(
                        file_dir, relative_results_dir, dist["dist"], act, init
                    )
                    data[dist["dist"]][act][init][name] = load_file(base_path, name, **dist)

    return data

def load_file(path, name, dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
    try:
        data_path = os.path.join(path, "{}_{}_{}_{}_{}_{}.npy".format(
            dist, std, diversity, prob_1, _lambda, name
        ))
        return np.load(data_path)
    except:
        try:
            data_path = os.path.join(path, "{}_{}_{}_{}_{}_{}.npz".format(
                dist, std, diversity, prob_1, _lambda, name
            ))
            return np.load(data_path)
        except:
            raise FileNotFoundError("can't find {}".format(path))