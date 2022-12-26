from __future__ import annotations

import csv
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Adam

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData

from jlo.drl_evolved_body.interaction_buffer import Buffer
from jlo.drl_evolved_body.actor_critic_network import Actor, ActorCritic, Critic, ObservationEncoder
from jlo.drl_evolved_body.config import LR_ACTOR, LR_CRITIC, PPO_CLIP_EPS, LR_ACTOR, LR_CRITIC, N_EPOCHS, \
    CRITIC_LOSS_COEFF, ENTROPY_COEFF, ACTOR_LOSS_COEFF

from torchsummary import summary


class RLcontroller(ActorController):
    _num_input_neurons: int
    _num_output_neurons: int
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
            self,
            actor_critic: Actor,
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
        self._actor_critic = actor_critic
        actor_params = [p for p in self._actor_critic.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self._actor_critic.critic.parameters() if p.requires_grad]
        self.actor_optimizer = Adam(actor_params, lr=LR_ACTOR, eps=1e-5)
        self.critic_optimizer = Adam(critic_params, lr=LR_CRITIC, eps=1e-5)
        # self.optimizer = Adam([p for p in self._actor_critic.parameters() if p.requires_grad])
        if from_checkpoint:
            checkpoint = torch.load("./last_checkpoint")
            self._iteration_num = checkpoint['iteration']
            self._actor_critic.load_state_dict(checkpoint['model_state'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        self._dof_ranges = dof_ranges
        # self.device = torch.device("cuda:0")
        # self._actor_critic.to(self.device)

    def get_dof_targets(self, observation) -> List[float]:
        """
        Get the target position for the motors of the body
        """
        action_prob, value, _, _ = self._actor_critic(observation)
        action = action_prob.sample()
        logp = action_prob.log_prob(action).sum(-1)
        return action, value, logp

    def train(self, buffer: Buffer):
        """
        Train the neural network used as controller
        args:
            buffer: replay buffer containing the data for the last timestep
        """

        print(f"\nITERATION NUM: {self._iteration_num + 1}")

        # learning rate decreases linearly
        lr_linear_decay(self.actor_optimizer, self._iteration_num, 100, LR_ACTOR)
        lr_linear_decay(self.critic_optimizer, self._iteration_num, 100, LR_CRITIC)

        self._iteration_num += 1
        buffer._compute_advantages()

        for epoch in range(N_EPOCHS):
            batch_sampler = buffer.get_sampler()

            ppo_losses = []
            val_losses = []
            losses = []

            for obs, val, act, logp_old, rew, adv, ret in batch_sampler:
                # normalize advantages and returns
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ret = (ret - ret.min())
                ret = ret / (ret.max() + 1e-8)

                # get value entropy and new log probability for the observations
                _, value, logp, entropy = self._actor_critic(obs, act)

                # compute ratio between new and old policy and losses
                ratio = torch.exp(logp - logp_old)
                obj1 = ratio * adv
                obj2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv  # ratio clipping
                print(
                    f"Percentage of clipped ratios: {int((abs(ratio) - 1 > PPO_CLIP_EPS).sum() / ratio.shape[0] * 100)}%")
                ppo_loss = -torch.min(obj1, obj2).mean()  # policy loss
                val_loss = (ret - value).pow(2).mean()  # value loss
                print(
                    f"[CRITIC LOSS]: {val_loss:.10f}     [ACTOR LOSS]: {ppo_loss:.10f}      [ENTROPY]: {entropy:.10f}")

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss = CRITIC_LOSS_COEFF * val_loss + ACTOR_LOSS_COEFF * ppo_loss - ENTROPY_COEFF * entropy
                loss.backward()
                ppo_losses.append(ppo_loss.item())
                val_losses.append(val_loss.item())
                losses.append(loss.item())

                self.actor_optimizer.step()
                self.critic_optimizer.step()

            print(
                f"EPOCH {epoch + 1} loss ppo:  {np.mean(ppo_losses):.5f}, loss val: {np.mean(val_losses):.5f}, final loss: {np.mean(losses):.5f}")
            print()
        state = {
            'iteration': self._iteration_num,
            'model_state': self._actor_critic.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
        }
        torch.save(state, "./last_checkpoint")

        # log statistics
        mean_rew = torch.mean(torch.mean(buffer.rewards, axis=0)).item()
        mean_val = torch.mean(torch.mean(buffer.values, axis=0)).item()
        with open('./statistics.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([mean_rew, mean_val])

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
    def deserialize(cls, data: StaticData) -> RLcontroller:
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
        critic = Critic(in_dim, out_dim)
        critic.load_state_dict(data["critic_state"])
        encoder = ObservationEncoder(in_dim)
        encoder.load_state_dict(data["encoder_state"])
        network = ActorCritic(in_dim, out_dim)
        network.actor = actor
        network.critic = critic
        network.encoder = encoder
        return RLcontroller(
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
