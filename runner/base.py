from abc import ABC, abstractmethod

import torch


class BaseRunner(ABC):

    def __init__(self, sys, agent):
        self.sys = sys
        self.agent = agent

    def set_batch_systems(self, nb_systems):
        param = self.agent.get_batch_systems(nb_systems)
        self.sys.set_parameters(param)
        self.sys.project_parameters()

        return param

    def set_system(self):
        param = self.agent.get_system()
        self.sys.set_parameters(param)
        self.sys.project_parameters()

        return param

    @abstractmethod
    def sample(self, nb_trajectories):
        """ sample a batch of trajectories (auto initializes the system)"""
        raise NotImplementedError

    def _sample(self, nb_trajectories):
        """ sample a batch of trajectories from an initialized system"""
        # batches
        state_batch = []
        dist_batch = []
        reward_batch = []
        action_batch = []

        # sample initial states
        states = self.sys.initial_state(nb_trajectories)
        state_batch.append(states)

        # generate the trajectory
        for t in range(0, self.sys.horizon):
            actions = self.agent(states)
            next_states, disturbances, reward, action = self.sys(states, actions)

            state_batch.append(next_states)
            dist_batch.append(disturbances)
            reward_batch.append(reward)
            action_batch.append(action)

            states = next_states

        return [torch.stack(l, dim=1) for l in [state_batch, dist_batch, reward_batch, action_batch]]

    @staticmethod
    def cumulative_reward(reward_batch):
        return torch.mean(torch.sum(reward_batch, dim=1), dim=0).item()
