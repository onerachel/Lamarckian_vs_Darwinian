import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler
from config import NUM_OBSERVATIONS, BUFFER_SIZE, BATCH_SIZE

class ReplayBuffer(object):
    """
    Replay buffer
    It stores observations, actions, values and rewards for each step of the simulation
    Used to create batches of data to train the controller 
    """
    def __init__(self, obs_dim, act_dim):
        self.observations = []
        for dim in obs_dim:
            self.observations.append(torch.zeros(BUFFER_SIZE, dim))

        self.next_observations = []
        for dim in obs_dim:
            self.next_observations.append(torch.zeros(BUFFER_SIZE, dim))

        self.actions = torch.zeros(BUFFER_SIZE, act_dim)
        self.rewards = torch.zeros(BUFFER_SIZE)
        self.logps = torch.zeros(BUFFER_SIZE)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.step = 0
        self.dim = 0

    def insert(self, obs, act, logp, rew, next_obs):
        """
        Insert a new step in the replay buffer
        Used when inserting the data of a new step for a single agent
        args:
            idx: the index of the agent to which the data is referred
            obs: observation at the current state
            act: action performed at the state
            logp: log probability of the action performed according to the current policy
            val: value of the state
            rew: reward received for performing the action
            next_obs: observation at the next state
        """
        if self.step >= BUFFER_SIZE:
            self.reset_step_count()

        if self.dim < BUFFER_SIZE:
            self.dim += 1

        for i, observation in enumerate(obs):
            self.observations[i][self.step,] = torch.from_numpy(observation)
        self.actions[self.step] = torch.tensor(act)
        self.rewards[self.step] = rew
        self.logps[self.step] = torch.tensor(logp)
        for i, observation in enumerate(next_obs):
            self.next_observations[i][self.step] = torch.from_numpy(observation)

        self.step += 1

    def get_sampler(self,):
        """
        Create a BatchSampler that divides the data in the buffer in batches 
        """
        dset_size = self.dim
        batch_size = BATCH_SIZE

        assert dset_size >= batch_size

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            batch_size,
            drop_last=True,
        )

        for idxs in sampler:
            obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, o in enumerate(self.observations):
                obs[i] = o.view(-1, self.obs_dim[i])[idxs]
            act = self.actions.view(-1, self.act_dim)[idxs]
            logp_old = self.logps.view(-1)[idxs]
            rew = self.rewards.view(-1)[idxs]
            next_obs = [[] for _ in range(NUM_OBSERVATIONS)]
            for i, o in enumerate(self.next_observations):
                next_obs[i] = o.view(-1, self.obs_dim[i])[idxs]
            yield obs, act, logp_old, rew, next_obs

    def sample(self):

        dset_size = self.dim
        batch_size = BATCH_SIZE

        assert dset_size >= batch_size
        idxs = np.random.randint(low=0, high=dset_size, size=batch_size)

        obs = [[] for _ in range(NUM_OBSERVATIONS)]
        for i, o in enumerate(self.observations):
            obs[i] = o.view(-1, self.obs_dim[i])[idxs]
        act = self.actions.view(-1, self.act_dim)[idxs]
        logp_old = self.logps.view(-1)[idxs]
        rew = self.rewards.view(-1)[idxs]
        next_obs = [[] for _ in range(NUM_OBSERVATIONS)]
        for i, o in enumerate(self.next_observations):
            next_obs[i] = o.view(-1, self.obs_dim[i])[idxs]

        return obs, act, logp_old, rew, next_obs


    def reset_step_count(self):
        self.step = 0
