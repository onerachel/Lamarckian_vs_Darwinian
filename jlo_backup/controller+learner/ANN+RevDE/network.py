from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        """
        Network that represents the actor 
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.action_layer = nn.Linear(32, act_dim)

    def forward(self, obs):
        pi_obs = self.obs_encoder(obs)
        action = self.action_layer(pi_obs)
        return action

class SingleObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Encoder for a single type of observation
        args:
            obs_dim: dimension of the observation in input to the newtork
        """
        super().__init__()
        dims = [obs_dim] + [32]
        
        self.encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layer = nn.Linear(in_features=dim_in, out_features=dim_out)
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)
            self.encoder.add_module(name=f"single_observation_encoder_{n}", module=layer)
            self.encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, obs):
        return self.encoder(obs)

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim: List[int]):
        """
        Full encoder: concatenate encoded observations and produce a 32-dim vector
        args:
            obs_dim: a list of the dimensions of all the observations to encode
        """
        super().__init__()
        self.encoders = torch.nn.ModuleList()
        self.obs_dim = obs_dim
        for obs_d in obs_dim:
            self.encoders.append(SingleObservationEncoder(obs_d))

        dims = [len(obs_dim) * 32] + [32]
        
        self.final_encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layer = nn.Linear(in_features=dim_in, out_features=dim_out)
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0)            
            self.final_encoder.add_module(name=f"final_observation_encoder_{n}", module=layer)
            self.final_encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, observations):
        if len(observations[0].shape) > 1:
            encoded_observations = torch.zeros(observations[0].shape[0], len(observations) * 32)
            for i, obs in enumerate(observations):
                encoded_observations[:,i * 32: i * 32 + 32] = self.encoders[i](obs)
        else:
            encoded_observations = torch.zeros(len(observations) * 32)
            for i, obs in enumerate(observations):
                encoded_observations[i * 32: i * 32 + 32] = self.encoders[i](obs)

        return self.final_encoder(encoded_observations)