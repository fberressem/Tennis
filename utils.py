import torch
import numpy as np
from collections import deque
import random

def copy_model(model_from, model_to, tau=1.0):
    """ Soft update of model weights weighted with tau 

    Params
    ======
        model_from(Model): model to copy from
        model_to(Model): model to copy to
        tau(float): weight factor for soft update 
    """
    for target_param, policy_param in zip(model_to.parameters(), model_from.parameters()):
        target_param.data.copy_((1-tau)*target_param.data + tau * policy_param.data)

class ReplayBuffer():
    """ Replaybuffer for usage in Agent.""" 
    def __init__(self, size, batchsize, device):
        """ Initialize a ReplayBuffer object 

        Params
        ======
            size(int): size of the replaybuffer
            batchsize(int): size of a sample from the replaybuffer
        """
        self.buffer = deque(maxlen=size)
        self.batchsize = batchsize
        self.device = device

    def append(self, state, action, reward, next_state, done):
        """ Append new experience 

        Params
        ======
            state(torch.Tensor): current state
            action(int): chosen action
            reward(float): observed reward
            next_state(torch.Tensor): next state
            done(bool): flag indicating terminal state
        """
        self.buffer.append([state, action, reward, next_state, done])

    def copy_last_trajectory(self, other, start, end=0):
        """ Copy the last trajectory into another replay buffer 

        Params
        ======
            other(ReplayBuffer): replay buffer to copy into
            start(int): backwards-index from which to start
            end(int): backwards-index at which to end
        """        
        for i in range(-min(start, len(self.buffer)), end):
            state, action, reward, next_state, done = self.buffer[i]
            other.append(state, action, reward, next_state, done)

    def sample(self):
        """ Returns sample of memories from replaybuffer """
        experiences = random.sample(self.buffer, k=min(self.batchsize, len(self.buffer)))
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """ Returns current length of replaybuffer """
        return self.buffer.__len__()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma        
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
