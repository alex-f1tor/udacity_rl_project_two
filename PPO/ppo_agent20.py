import numpy as np
import random
import copy
from collections import namedtuple, deque
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model import Actor, Critic
from torch.distributions import Normal
import torch
import torch.nn.functional as F
import torch.optim as optim
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters  
LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate critic 

PPO_UPDATE_PERIOD = 8



class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.clip_param = 0.8
        self.training_step = 0
        self.counter = 0
        self.training_step = 0
        self.memory_capacity = 100000
        self.batch_size = BATCH_SIZE
        self.memory = []
        self.gamma = GAMMA

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, 3*random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            (mu, sigma) = self.actor_local(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-1.0, 1.0).shape
        #print('act_shape', action.shape)
        #print('act_log_prob', action_log_prob.shape)
        return action.cpu().numpy(), action_log_prob.cpu().numpy()
    
    
    def get_value(self, state):

        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            state_value = self.critic_local(state)
        return state_value.item()
    
    def store_transition(self, transition):
        self.memory.append(transition)
        self.counter += 1
        return self.counter % self.memory_capacity == 0
    
    def reset(self):
        self.noise.reset()

    def learn(self, epoch_index):
        self.training_step += 1
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state = torch.tensor([t.state for t in self.memory], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.memory], dtype=torch.float).to(device)
        reward = torch.tensor([t.reward for t in self.memory], dtype=torch.float).to(device).unsqueeze(-1)
        next_state = torch.tensor([t.next_state  for t in self.memory], dtype=torch.float).to(device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.memory], dtype=torch.float).to(device)
        
        #print('reward full pure shape', reward.shape)
        
        reward = (reward - reward.mean())/(reward.std()+0.00001)
        
        #print('reward full scaled shape', reward.shape)
        
        target_v = []
        
        for mem_index in range(len(state)): 
            with torch.no_grad():
                target_v.append(reward[mem_index] + self.gamma * self.critic_local(next_state[mem_index]))
        
        target_v =  torch.stack(target_v).to(device)
        #print('target_v.shape',target_v.shape)
        
        advantage = []
        for mem_index in range(len(state)): 
            with torch.no_grad():
                advantage.append(target_v[mem_index]-self.critic_local(state[mem_index]))
        advantage =  torch.stack(advantage).to(device)
        #print('advantage shape', advantage.shape)
        
        
                                
        for i in range(PPO_UPDATE_PERIOD):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), BATCH_SIZE, False):
                #states, actions, old_action_probs, rewards, next_states, dones = self.memory.sample()
                #print('index', index)
                #print('state_indexed_shape', state[index].shape)
                (mu, sigma) = self.actor_local(state[index])
                dist = Normal(mu, sigma)
                
                #print('action full size',action.shape)
                #print('action indexed',action[index])
                action_prob = dist.log_prob(action[index])
                
                #print('action_prob', action_prob.shape, '\nold_action_prob shape', old_action_log_prob.shape)
                
                ratio = torch.exp(action_prob - old_action_log_prob[index])
                
                #print('ratio_shape',ratio.shape)
                #print('ratio_shape', advantage[index].shape)

                                
                L_left = ratio*advantage[index]
                L_right = torch.clamp(ratio, 1-self.clip_param-0.5, 1+self.clip_param+0.5)*advantage[index]

                #update actor network
                action_loss = -torch.min(L_left, L_right).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
                self.actor_optimizer.step()

                #update critic optimizer
                value_loss = F.smooth_l1_loss(self.critic_local(state[index]), target_v[index])
                if i==5:
                    print('val_loss', value_loss)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
                self.critic_optimizer.step()
                
            print('PPO step', i)
        
        if (epoch_index%10==0)&(epoch_index>1):
            del self.memory[:]  

