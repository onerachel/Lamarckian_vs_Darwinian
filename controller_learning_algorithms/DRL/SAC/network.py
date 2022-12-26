from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from config import ACTION_CONSTRAINT

from typing import List

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        """
        Network that approximates the Q-function
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """

        super().__init__()
        self.qval_encoder = InputEncoder(obs_dim + (act_dim,))
        self.qval_layer = nn.Linear(32,1)

    def forward(self, obs, action):
        state = obs.copy()
        state.append(action)
        encoded_inputs = self.qval_encoder(state)
        return self.qval_layer(encoded_inputs)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        """
        Network that represents the actor 
        args:
            obs_dim: dimension of the observation in input to the newtork
            act_dim: number of actuators in output
        """
        super().__init__()
        self.pi_encoder = InputEncoder(obs_dim)
        self.mean_layer = nn.Linear(32, act_dim)
        self.std_layer = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, action=None):
        pi_obs = self.pi_encoder(obs)
        mean = self.mean_layer(pi_obs)
        std = torch.exp(self.std_layer)
        action_prob = Normal(mean, std)
        if action is None:
            return action_prob
        else:
            logp = action_prob.log_prob(action).sum(-1)
            entropy = action_prob.entropy().mean()
            return action_prob, logp, entropy


class SingleInputEncoder(nn.Module):
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
            self.encoder.add_module(name=f"single_observation_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, obs):
        return self.encoder(obs)

class InputEncoder(nn.Module):
    def __init__(self, input_dim: List[int]):
        """
        Full encoder: concatenate encoded inputs and produce a 32-dim vector
        args:
            obs_dim: a list of the dimensions of all the inputs to encode
        """
        super().__init__()
        self.encoders = torch.nn.ModuleList()
        self.input_dim = input_dim
        for input_d in input_dim:
            self.encoders.append(SingleInputEncoder(input_d))

        dims = [len(input_dim) * 32] + [32]
        
        self.final_encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.final_encoder.add_module(name=f"final_input_encoder_{n}", module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.final_encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, input):
        if len(input[0].shape) > 1:
            encoded_inputs = torch.zeros(input[0].shape[0], len(input) * 32)
            for i, obs in enumerate(input):
                encoded_inputs[:,i * 32: i * 32 + 32] = self.encoders[i](obs)
        else:
            encoded_inputs = torch.zeros(len(input) * 32)
            for i, obs in enumerate(input):
                encoded_inputs[i * 32: i * 32 + 32] = self.encoders[i](obs)

        return self.final_encoder(encoded_inputs)