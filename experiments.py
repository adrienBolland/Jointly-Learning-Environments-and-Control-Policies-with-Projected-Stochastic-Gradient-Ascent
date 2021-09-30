import os
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import torch

import initialize
import utils


def fit(sys, agent, device, path, logger, algo, seed_val=None, save_agent=False, reset_agent=True):
    """ trains the agent in the system """
    # get the path for the logger and the saves
    log_path = utils.get_log_path(path["model_name"], path["logdir"])

    # launch the experiment
    try:
        _fit(sys, agent, device, log_path, logger, algo, seed_val, save_agent, reset_agent)
    except Exception as e:
        if log_path is not None:
            print("-An exception occurred-")
            print(e)
            utils.save_error(log_path, e)

    return log_path


def _fit(sys, agent, device, log_path, logger, algo, seed_val, save_agent, reset_agent):
    """ trains the agent in the system - directly work on the correct path"""
    # set the seed if requested
    if seed_val is not None:
        seed(sys, agent, device, seed_val)

    # reset the parameters of the agent using the new seed
    if reset_agent:
        agent.reset_parameters()

    # initialize the logger
    log_writer = initialize.get_logger(log_path, **logger)

    # initialize the algo
    algorithm = initialize.get_algo(sys, agent, **algo)

    # fit the agent with the algo
    algorithm.fit(log_writer, device)

    # save the agent if requested
    if save_agent:
        save(sys, agent, device, log_path)


def simulate(sys, agent, device, runner, nb_simulations, render=False):
    """ performs simulations of an agent """
    cum_r, _ = _simulate(sys, agent, device, runner, nb_simulations, render)

    print("Average cumulative reward : ", cum_r)


def _simulate(sys, agent, device, runner, nb_simulations, render):
    sim_runner = initialize.get_runner(sys, agent, **runner)

    with torch.no_grad():
        state_batch, dist_batch, reward_batch, action_batch, param_batches = sim_runner.sample(nb_simulations)
        cum_r = sim_runner.cumulative_reward(reward_batch)

        if render:
            sys.render(state_batch, action_batch, dist_batch, reward_batch, nb_simulations)

    return cum_r, param_batches.view(-1, param_batches.shape[-1]).mean(dim=0).tolist()


def simulate_batch(sys, agent, device, runner, nb_simulations, dirs_batch):
    """ performs simulations of a series of environment """
    rewards = []
    parameters = []
    for path in utils.list_run_dirs(dirs_batch):
        load(sys, agent, device, path)
        rw, pr = _simulate(sys, agent, device, runner, nb_simulations, False)

        rewards.append(rw)
        parameters.append(pr)

    rewards = torch.tensor(rewards)
    parameters = torch.tensor(parameters)

    print("Max expected return : ", torch.max(rewards))
    print("Min expected return : ", torch.min(rewards))

    print("Average expected return : ", torch.mean(rewards))
    print("Median expected return : ", torch.median(rewards))

    print("Std expected return : ", rewards.std())
    print("Std+ expected return : ",
          torch.mean((rewards[rewards >= torch.mean(rewards)] - torch.mean(rewards)).pow(2)).sqrt())
    print("Std- expected return : ",
          torch.mean((rewards[rewards < torch.mean(rewards)] - torch.mean(rewards)).pow(2)).sqrt())

    print("Average parameter : ", torch.mean(parameters, dim=0))
    print("Std parameter : ", torch.std(parameters, dim=0))


def load(sys, agent, device, path):
    """ loads the agent models """
    agent.load(path)


def save(sys, agent, device, path):
    """ saves the agent models """
    agent.save(path)


def seed(sys, agent, device, value):
    """ sets the seeds of the different sources of uncertainty """
    utils.set_seeds(value)


def grid(sys, agent, device, path, logger, algo, nb_workers, grid_size, nb_run, nb_processes=1, save_agent=False):
    """ performs a grid search """
    raise NotImplementedError


def threads(sys, agent, device, nb_thread):
    """ set the number of cpu threads of torch """
    torch.set_num_threads(nb_thread)


def avg_fit(sys, agent, device, path, logger, algo, nb_run, nb_processes=1, seed_val=None, save_agent=False):
    """ trains the agent in the system several times """
    # get the path
    logdir = path["logdir"]
    model_name = path["model_name"]
    v = utils.get_version(model_name, logdir)
    log_path = os.path.join(logdir, f"{model_name}-v{v}")

    # build the argument list
    argument_list = []
    for run_id in range(nb_run):
        if seed_val is not None:
            seed_val += 1

        # build the argument of _fit
        kwargs = {"sys": deepcopy(sys),
                  "agent": deepcopy(agent),
                  "device": None,
                  "log_path": os.path.join(log_path, f"run-r{run_id}"),
                  "logger": logger,
                  "algo": algo,
                  "seed_val": seed_val,
                  "save_agent": save_agent,
                  "reset_agent": True}

        # add the argument to the list
        argument_list.append(kwargs)

    # create pools
    with Pool(processes=nb_processes) as pool:
        pool.map(partial(_direct_arg_fit), argument_list)

    return log_path


def _direct_arg_fit(arg_dict):
    return _fit(**arg_dict)


EXP_DICT = {"fit": fit,
            "simulate": simulate,
            "simulate_batch": simulate_batch,
            "load": load,
            "save": save,
            "seed": seed,
            "threads": threads,
            "avg_fit": avg_fit}
