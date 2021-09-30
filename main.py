import argparse

import algo
import experiments
import initialize
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="script launching experiments.")

    parser.add_argument("-conf", "--config_file", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-vv", "--very_verbose", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-th", "--threads", type=int)
    return vars(parser.parse_args())


if __name__ == "__main__":
    # get the arguments
    parsed_args = parse_args()

    # get the config file
    config = utils.load_json(parsed_args["config_file"])

    # set the algorithms as verbose if required
    algo.VERY_VERBOSE = parsed_args["very_verbose"]
    algo.VERBOSE = parsed_args["verbose"] or algo.VERY_VERBOSE

    # check if the device exists
    device = parsed_args["device"]
    
    # create the system and the agent
    sys = initialize.get_system(**config["system"])
    agent = initialize.get_agent(sys, **config["agent"])

    # set the number of threads
    nb_threads = parsed_args["threads"]
    if nb_threads is not None:
        experiments.EXP_DICT["threads"](sys, agent, device, nb_threads)

    # set seed if required
    seed = parsed_args["seed"]
    if seed is not None:
        experiments.EXP_DICT["seed"](sys, agent, device, seed)

    # launch the experiments
    experiments_dict = experiments.EXP_DICT
    for exp_name in config["experiment"]:
        exp_f_name = exp_name.split(".").pop(0)
        if exp_f_name not in experiments_dict:
            raise Exception(f"Unknown experiment requested: {exp_f_name}")

        exp_f = experiments_dict[exp_f_name]
        log_path = exp_f(sys, agent, device, **config["experiment"][exp_name])

        # if a path is provided, dum the config in this path
        if log_path is not None:
            utils.save_config(log_path, config)
