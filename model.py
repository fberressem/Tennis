import torch
import torch.nn.functional as F
import torch.nn as nn

class ActorModel(nn.Module):
    """Neural network for usage as actor in agent"""
    def __init__(self, param_dict={}):
        """ Initialize an ActorModel object.

        Params
        ======
           param_dict(dictionary): contains size-information
        """
        super().__init__()

        input_size = param_dict.get("input_size", 33)
        self.output_size = param_dict.get("output_size", 4)
        self.batch_norm = param_dict.get("batch_norm", False)
        hn = param_dict.get("hn", [128, 128, 64, 32])

        hn = [input_size] + hn + [self.output_size]

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

        self.hidden = nn.ModuleList()
        for k in range(len(hn)-1):
            self.hidden.append(nn.Linear(hn[k], hn[k+1]))

    def forward(self, x):
        """ Defines forward pass. Returns proposed action given state x.

        Params
        ======
           x(torch.tensor): current state
        """

        if self.batch_norm:
            x = self.bn(x)

        for k in range(len(self.hidden)-1):
            x = F.relu(self.hidden[k](x))
        x = F.tanh(self.hidden[-1](x))
        return x



class CriticModel(nn.Module):
    """Neural network for usage as critic in agent"""
    def __init__(self, param_dict={}):
        """ Initialize a CriticModel object.

        Params
        ======
           param_dict(dictionary): contains size-information and stage at which action should be concatenated
        """
        super().__init__()

        state_size = param_dict.get("state_size", 33)
        self.action_size = param_dict.get("action_size", 4)
        self.batch_norm = param_dict.get("batch_norm", False)
        hn = param_dict.get("hn", [128, 128, 64, 32])
        self.concat_stage = param_dict.get("concat_stage", 0)   # parameter to set at which state the action should be concatenated

        hn = [state_size] + hn + [1]

        if self.batch_norm:
            self.bn_state = nn.BatchNorm1d(state_size)
            self.bn_action = nn.BatchNorm1d(self.action_size)

        self.hidden = nn.ModuleList()
        for k in range(len(hn)-1):
            current_size = hn[k] + (self.action_size if k == self.concat_stage else 0)
            self.hidden.append(nn.Linear(current_size, hn[k+1]))

    def forward(self, state, action):
        """ Defines forward pass. Returns action-value of given set of state and action.

        Params
        ======
           state(torch.tensor): current state
           action(torch.tensor): proposed action
        """
        if self.batch_norm:
            state = self.bn_state(state)
            action = self.bn_action(action)

        x = state

        for k in range(0, len(self.hidden)-1):
            if self.concat_stage == k:
                x = torch.cat((x, action), dim = 1)
            x = F.relu(self.hidden[k](x))
        x = self.hidden[-1](x)
        return x



