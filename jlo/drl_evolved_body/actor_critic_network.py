import enum
from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from typing import List


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        """
        Neural Network (Gaussian action distribution) that represents the actor
        args:
            obs_dim: dimension of the observation in input to the network
            act_dim: number of actuators in output
        """
        super().__init__()
        self.pi_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.mean_layer = nn.Linear(32, act_dim)
        self.std_layer = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        pi_obs = self.pi_encoder(obs)
        mean = self.mean_layer(pi_obs)
        std = torch.exp(self.std_layer)
        action_prob = Normal(mean, std)
        return action_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Network that represents the Critic (Q-value prediction)
        args:
            obs_dim: dimension of the observation in input to the network
        """
        super().__init__()
        self.val_encoder = ObservationEncoder(obs_dim=obs_dim)
        self.critic_layer = nn.Linear(32, 1)

    def forward(self, obs):
        val_obs = self.val_encoder(obs)
        return self.critic_layer(val_obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: List[int], act_dim: int):
        """
        args:
            obs_dim: dimension of the observation in input to the network
            act_dim: number of actuators in output
        """
        super().__init__()
        self.actor = Actor(obs_dim=obs_dim, act_dim=act_dim)
        self.critic = Critic(obs_dim=obs_dim)

    def forward(self, obs, action=None):
        action_prob = self.actor(obs)
        value = self.critic(obs)
        if action == None:
            return action_prob, value, None, None
        else:
            logp = action_prob.log_prob(action).sum(-1)
            entropy = action_prob.entropy().mean()
            return action_prob, value, logp, entropy


class SingleObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int):
        """
        Encoder for a single type of observation
        args:
            obs_dim: dimension of the observation in input to the network
        """
        super().__init__()
        dims = [obs_dim] + [32]

        self.encoder = nn.Sequential()
        for n, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.encoder.add_module(name=f"single_observation_encoder_{n}",
                                    module=nn.Linear(in_features=dim_in, out_features=dim_out))
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
            self.final_encoder.add_module(name=f"final_observation_encoder_{n}",
                                          module=nn.Linear(in_features=dim_in, out_features=dim_out))
            self.final_encoder.add_module(name=f'tanh_{n}', module=nn.Tanh())

    def forward(self, observations):
        if len(observations[0].shape) > 1:
            encoded_observations = torch.zeros(observations[0].shape[0], len(observations) * 32)
            for i, obs in enumerate(observations):
                encoded_observations[:, i * 32: i * 32 + 32] = self.encoders[i](obs)
        else:
            encoded_observations = torch.zeros(len(observations) * 32)
            for i, obs in enumerate(observations):
                encoded_observations[i * 32: i * 32 + 32] = self.encoders[i](obs)

        return self.final_encoder(encoded_observations)
