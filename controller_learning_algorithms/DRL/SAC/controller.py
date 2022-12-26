from __future__ import annotations
from asyncore import write
import csv
import logging

from typing import List
import copy

import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Adam
import torch.nn.functional as F

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData

from replay_buffer import ReplayBuffer
from network import Actor, InputEncoder, SoftQNetwork
from config import GAMMA, LR_ACTOR, LR_CRITIC, NUM_ITERATIONS, LR_ACTOR, LR_CRITIC, N_EPOCHS, INIT_TEMPERATURE, LR_ALPHA, NUM_PARALLEL_AGENT, NUM_STEPS, TAU

class SACcontroller(ActorController):
    _num_input_neurons: int
    _num_output_neurons: int
    _dof_ranges: npt.NDArray[np.float_]
    _file_path: str

    def __init__(
        self,
        file_path: str,
        actor: Actor,
        q1: SoftQNetwork,
        q2: SoftQNetwork,
        dof_ranges: npt.NDArray[np.float_],
        from_checkpoint: bool = False,
    ):
        """
        The controller for an agent
        args:
            actor_critic: Neural Network controlling the agents
            dof_ranges: value range for the agent motors
            from_checkpoint: if True, resumes training from the last checkpoint
        """
        self._iteration_num = 0
        self._actor = actor
        self._q1 = q1
        self._q2 = q2
        self._q1_target = copy.deepcopy(q1)
        self._q2_target = copy.deepcopy(q2)
        self._log_alpha = torch.tensor(np.log(INIT_TEMPERATURE))
        self._log_alpha.requires_grad = True
        self._target_entropy = - (len(dof_ranges))

        actor_params = [p for p in self._actor.parameters() if p.requires_grad]
        q1_params = [p for p in self._q1.parameters() if p.requires_grad]
        q2_params = [p for p in self._q2.parameters() if p.requires_grad]

        self.actor_optimizer = Adam(actor_params, lr=LR_ACTOR)
        self.q1_optimizer = Adam(q1_params, lr=LR_CRITIC)
        self.q2_optimizer = Adam(q2_params, lr=LR_CRITIC)
        self.log_alpha_optimizer = Adam([self._log_alpha], lr=LR_ALPHA)

        self._file_path = file_path
        if from_checkpoint:
            checkpoint = torch.load(self._file_path + "/last_checkpoint")
            self._iteration_num = checkpoint['iteration']
            self._actor.load_state_dict(checkpoint['actor_state'])
            self._q1.load_state_dict(checkpoint['q1_state'])
            self._q2.load_state_dict(checkpoint['q2_state'])
            self._log_alpha = checkpoint['log_alpha']
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state'])
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state'])
            self.log_alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state'])
        self._dof_ranges = dof_ranges

    def get_dof_targets(self, observation) -> List[float]:
        """
        Get the target position for the motors of the body
        """
        action_prob = self._actor(observation)
        action = action_prob.sample()
        logp = action_prob.log_prob(action).sum(-1)
        return action, logp
    
    def train(self, buffer: ReplayBuffer):
        """
        Train the neural network used as controller
        args:
            buffer: replay buffer containing the data for the last timesteps
        """

        logging.info(f"ITERATION NUM: {self._iteration_num + 1}")

        # learning rate decreases linearly
        #lr_linear_decay(self.actor_optimizer, self._iteration_num, NUM_ITERATIONS, LR_ACTOR)
        #lr_linear_decay(self.q1_optimizer, self._iteration_num, NUM_ITERATIONS, LR_CRITIC)
        #lr_linear_decay(self.q2_optimizer, self._iteration_num, NUM_ITERATIONS, LR_CRITIC)

        self._iteration_num += 1

        for epoch in range(N_EPOCHS):

            actor_losses = []
            q1_losses = []
            q2_losses = []
            alpha_losses = []
            
            obs, act, logp_old, rew, next_obs = buffer.sample()

            action_prob = self._actor(next_obs)
            next_action = action_prob.sample()
            logp = action_prob.log_prob(next_action).sum(-1)
            target_q1 = self._q1_target(next_obs, next_action)
            target_q2 = self._q2_target(next_obs, next_action)
            target_v = torch.min(target_q1, target_q2) - (self.alpha().detach() * logp).unsqueeze(dim=-1)
            target_q = rew.unsqueeze(dim=-1) + GAMMA * target_v
            target_q = target_q.detach()

            #get current q estimates
            current_q1 = self._q1(obs, act)
            current_q2 = self._q2(obs, act)
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)

            action_dist = self._actor(obs)
            action = action_dist.sample()
            logp = action_prob.log_prob(action).sum(-1)
            actor_q1 = self._q1(obs, action)
            actor_q2 = self._q2(obs, action)
            actor_q = torch.min(actor_q1, actor_q2)
            actor_loss = (self.alpha().detach() * logp - actor_q).mean()

            alpha_loss = (self.alpha() * (-logp - self._target_entropy).detach()).mean()

            self.actor_optimizer.zero_grad()
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            self.log_alpha_optimizer.zero_grad()
            actor_loss.backward()
            q1_loss.backward()
            q2_loss.backward()
            alpha_loss.backward()
            actor_losses.append(actor_loss.item())
            q1_losses.append(q1_loss.item())
            q2_losses.append(q2_loss.item())
            alpha_losses.append(alpha_loss.item())

            self.actor_optimizer.step()
            self.q1_optimizer.step()
            self.q2_optimizer.step()
            self.log_alpha_optimizer.step()

            # update target networks
            self._q1_target = soft_copy(self._q1_target, self._q1, TAU)
            self._q2_target = soft_copy(self._q2_target, self._q2, TAU)

            logging.info(f"EPOCH {epoch + 1} actor loss:  {np.mean(actor_losses):.5f}, q1 loss: {np.mean(q1_losses):.5f}, q2 loss: {np.mean(q2_losses):.5f}, alpha loss: {np.mean(alpha_losses):.5f}")

        state = {
            'iteration': self._iteration_num,
            'actor_state': self._actor.state_dict(),
            'q1_state': self._q1.state_dict(),
            'q2_state': self._q2.state_dict(),
            'log_alpha': self._log_alpha,
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'q1_optimizer_state': self.q1_optimizer.state_dict(),
            'q2_optimizer_state': self.q2_optimizer.state_dict(),
            'alpha_optimizer_state': self.log_alpha_optimizer.state_dict(),
        }
        torch.save(state, self._file_path + "/last_checkpoint")

        # log statistics
        mean_rew = torch.mean(torch.mean(buffer.rewards[buffer.step-(NUM_PARALLEL_AGENT*NUM_STEPS):buffer.step], axis=0)).item()
        with open(self._file_path + '/statistics.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mean_rew,])

    def alpha(self):
        return self._log_alpha.exp()

    # TODO
    def step(self, dt: float):
        return

    # TODO
    def serialize(self) -> StaticData:
        return {
            "num_input_neurons": self._num_input_neurons,
            "num_output_neurons": self._num_output_neurons,
            "dof_ranges": self._dof_ranges.tolist(),
        }

    # TODO
    @classmethod
    def deserialize(cls, data: StaticData) -> SACcontroller:
        if (
            not type(data) == dict
            or not "actor_state" in data
            or not "critic_state" in data
            or not "encoder_state" in data
            or not "num_input_neurons" in data
            or not "num_output_neurons" in data
            or not "dof_ranges" in data
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        in_dim = data["num_input_neurons"]
        out_dim = data["num_output_neurons"]
        actor = Actor(in_dim, out_dim)
        actor.load_state_dict(data["actor_state"])
        critic = SoftQNetwork(in_dim, out_dim)
        critic.load_state_dict(data["critic_state"])
        encoder = InputEncoder(in_dim)
        encoder.load_state_dict(data["encoder_state"])
        network = Actor(in_dim, out_dim)
        network.actor = actor
        network.critic = critic
        network.encoder = encoder
        return SACcontroller(
            network,
            np.array(data["dof_ranges"]),
        )

def lr_linear_decay(optimizer, iter, total_iters, initial_lr):
    """
    Decrease the learning rate linearly
    """
    lr = initial_lr - (initial_lr * (iter / float(total_iters)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def soft_copy(target, network, tau):
    for target_param, param in zip(target.parameters(), network.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    return target