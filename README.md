# Jointly Learning Environments and Control Policies with Projected Stochastic Gradient Ascent

This repository implements a library for joint design and control of environments.
More details on the theoretical background can be found in the following article:

[***Bolland, A., Boukas, I., Berger, M., & Ernst, D. (2021). Jointly Learning Environments and Control Policies with Projected Stochastic Gradient Ascent. arXiv preprint arXiv:2006.01738v3.***](https://arxiv.org/abs/2006.01738v3)

## Code organisation
The optimization of an environment and its control policy is divided into three main parts.

First, the environment to be optimized must be provided.
Such an environment implements the base abstract class provided in `system/base.py`. 
Three environments are already implemented in the library:
* ***DroneTrajectory*** models the dynamics of a drone that shall fly on an elliptical trajectory;
* ***ComplexMGSystem*** implements the dynamics of a solar off-grid microgrid in which we operate a generator and a battery for supplying the electricity consumption of a load at low cost;
* ***MSDSystem*** provides an implementation of a mass-spring-damper system that shall be maintained at a reference position.

Second, agents implementing the abstract class from `agent/base.py` provide the behaviour to adopt in an environment.
They can be separated into rule-based agents (`agent/rules`) and trainable agents (`agent/trainable`).
The latter agents depend on parametric models where some parameters shall be learned with an algorithm.
Two noteworthy agents provided in `agent/trainable/DESGA` are:
* ***DSSAAgent*** for Deterministic System and Stochastic control Agent;
* ***S3AAgent*** for Stochastic System and Stochastic control Agent.

The first agents select deterministically an environment and stochastically the actions to perform in the environment.
The second selects both decisions stochastically. 
These two agents depend on parametric models implemented in `model/decision` and `model/investement` for control and design decisions, respectively.

Finally, trainable agents can be trained in environments using algorithms implementing the abstract class from `algo/base.py`.
Two algorithm for jointly designing and controlling environments are implemented:
* ***ReinforceDESGA*** implementing Direct Environment and Policy Search (DEPS) proposed by [Bolland et al, 2021](https://arxiv.org/abs/2006.01738v3);
* ***PPOJODC*** implementing Joint Optimization of Design and Control (JODC) proposed by [Schaff et al, 2019](https://arxiv.org/abs/1801.01432).

A tutorial notebook using DEPS for optimizing the MSD environment is provided in `example/msd_ex1.ipynb`.

## Launching experiments
Out-of-the-box experiments can be launched from json configuration files with the following command:

> python main.py -conf *config_file.json* [-s *seed*] [-v|-vv]

Configuration files for the three environments are provided in `config_drone`, `config_cmg`, and `config_msd`.