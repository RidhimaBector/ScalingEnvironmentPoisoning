import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from attack.util import *
from attack.random_process import OrnsteinUhlenbeckProcess
from algorithms.algorithm import Algorithm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# Implementation: https://github.com/ghliu/pytorch-ddpg


criterion = nn.MSELoss()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



class Actor(nn.Module):
    """Neural network for the actor (policy).

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        max_action (float): Maximum action value
        hidden_dims (tuple): Dimensions of hidden layers (default: (400, 300))
        init_w (float): Initial weight range for final layer (default: 3e-3)
    """
    def __init__(self, state_dim, action_dim, max_action,
                 hidden_dims=(400, 300), init_w=3e-3):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)

        self.max_action = max_action
        self.init_weights(init_w)

    def init_weights(self, init_w):
        # Initialize hidden layers with fan-in scaling
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = fanin_init(layer.weight.data.size())

        # Initialize final layer uniformly
        self.output_layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.hidden_layers(state)
        return self.max_action * torch.tanh(self.output_layer(x))


class Critic(nn.Module):
    """Neural network for the critic (Q-function).

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        hidden_dims (tuple): Dimensions of hidden layers (default: (400, 300))
        init_w (float): Initial weight range for final layer (default: 3e-3)
    """
    def __init__(self, state_dim, action_dim,
                 hidden_dims=(400, 300), init_w=3e-3):
        super().__init__()

        # First layer processes state only
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])

        # Remaining layers process state+action
        layers = []
        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.state_layer.weight.data = fanin_init(self.state_layer.weight.data.size())

        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = fanin_init(layer.weight.data.size())

        self.output_layer.weight.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = F.relu(self.state_layer(state))
        x = torch.cat([x, action], dim=1)
        x = self.hidden_layers(x)
        return self.output_layer(x)


class DDPG(Algorithm):
    """Deep Deterministic Policy Gradient implementation.

    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        max_action (float): Maximum action value
        actor_kwargs (dict): Arguments for Actor network
        critic_kwargs (dict): Arguments for Critic network
        buffer_size (int): Size of replay buffer
        batch_size (int): Size of training batch
        discount (float): Discount factor gamma
        tau (float): Target network update rate
        actor_lr (float): Learning rate for actor
        critic_lr (float): Learning rate for critic
        exploration_noise (dict): Parameters for exploration noise
        device (torch.device): Device to use for tensor operations
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        actor_kwargs={},
        critic_kwargs={},
        buffer_size=1e6,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        exploration_noise={'theta': 0.15, 'mu': 0, 'sigma': 0.2},
        device=None
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action, **actor_kwargs).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, int(buffer_size))

        # Initialize noise process for exploration
        self.noise = OrnsteinUhlenbeckProcess(
            size=action_dim, **exploration_noise
        )

        # Save parameters
        self.max_action = max_action
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

    @torch.no_grad()
    def select_action(self, state, evaluate=False):
        """Select action given current state."""
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().numpy()

        if not evaluate:
            noise = self.noise.sample()
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def train(self, batch=None):
        """Update policy and value networks using sampled batch."""
        # Sample from replay buffer if batch not provided
        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size)

        self.is_training = True
        for i_atk_n_epoch in range(atk_n_epoch):

            loss_critic = 0.0
            loss_actor = 0.0
            for i_atk_n_batch in range(atk_n_batch):

                # Sample batch
                state_batch, action_batch, next_state_batch, \
                reward_batch, terminal_batch = replay_buffer.sample(self.batch_size) #self.memory.sample_and_split(self.batch_size)

                # Prepare for the target q batch
                next_q_values = self.critic_target(
                    to_tensor(next_state_batch, volatile=True),
                    self.actor_target(to_tensor(next_state_batch, volatile=True)))
                next_q_values.volatile=False

                target_q_batch = to_tensor(reward_batch) + \
                    self.discount*to_tensor(terminal_batch.astype(np.float64))*next_q_values

                # Critic update
                self.critic.zero_grad()

                q_batch = self.critic( to_tensor(state_batch), to_tensor(action_batch) )

                value_loss = criterion(q_batch, target_q_batch)
                value_loss.backward()
                self.critic_optim.step()

                # Actor update
                self.actor.zero_grad()

                policy_loss = -self.critic(
                    to_tensor(state_batch),
                    self.actor(to_tensor(state_batch))
                )

                policy_loss = policy_loss.mean()
                policy_loss.backward()
                self.actor_optim.step()

                # Target update
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

                loss_critic += value_loss.item()
                loss_actor += policy_loss.item()

            ddpg_loss.append([i_episode, i_atk_n_epoch, loss_critic/atk_n_batch, loss_actor/atk_n_batch])

        return ddpg_loss


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optim.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=torch.device('cpu')))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=torch.device('cpu')))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
        self.actor_optim.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
        self.actor_target = copy.deepcopy(self.actor)


    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def reset(self):
        pass

    def update(self, state, action, reward, next_state, done):
        pass
