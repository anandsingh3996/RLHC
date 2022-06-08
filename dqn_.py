=========================================
from lib import dqn_model
import grid2op
from grid2op import make
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
from   my_agent import MyAgent
import sys
import os, copy
import time
import random
import numpy as np
import copy
import collections


env = make("l2rpn_neurips_2020_track1_small")
obs = env.reset()
DEFAULT_ENV_NAME = make("l2rpn_neurips_2020_track1_small")

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 100

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
CLIP_GRAD     = 0.1


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, action, exp_buffer):
        this_directory_path = '../'
        
        self.env = env
        self.exp_buffer = exp_buffer
        self.action     = action
        self.action_space = env.action_space
        self.PPO_Agent = MyAgent(env.action_space, this_directory_path)
        self.PPO_Agent.load(this_directory_path)
        print('PPO Agent loaded')
        self._reset()

    def _reset(self):
        self.obs            = self.env.reset()
        # observation shape 
        chosen    =  list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        chosen    += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        chosen    += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        chosen    =  np.asarray(chosen, dtype=np.int32) - 1 
        # start sampling
        self.state = np.asarray(obs.to_vect()[chosen])  
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        all_actions, all_rho = [], []
        done_reward = None
        print("self.obs", self.obs.rho)
        self.state = np.asarray(self.obs.to_vect()[chosen])  
        # sample ppo model action 
        topo_act = self.PPO_Agent.act(self.obs, None, None)
        
        sim_obs, sim_reward, sim_done, sim_info = self.obs.simulate(topo_act)
        if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
            all_actions.append(copy.deepcopy(topo_act))
            all_rho.append(np.max(sim_obs.rho))
        

        if self.state[654:713].max() >= 0.97:
            if np.random.random() < epsilon:
                array = list(np.arange(0,len(self.action)))
                random.shuffle(array)
                
                for action in array:
                    act_    = self.action[action]  
                    sim_obs, sim_reward, sim_done, sim_info = self.obs.simulate(act_)
                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        act = copy.deepcopy(act_)
                        all_actions.append(copy.deepcopy(act_))
                        all_rho.append(np.max(sim_obs.rho))
                        break
                    else:
                        continue
            
            else:
                state_a = np.array([self.state], copy=False)
                state_v = torch.FloatTensor(state_a).to(device)
                q_vals_v = net(state_v)
                sorted_idx = np.argsort(q_vals_v)[::-1]
                
                for act_v in sorted_idx:
                    action = int(act_v)
                    act_   = self.action[action]
                    sim_obs, sim_reward, sim_done, sim_info = self.obs.simulate(act_)
                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        act = copy.deepcopy(act_)
                        all_actions.append(copy.deepcopy(topo_act))
                        all_rho.append(np.max(sim_obs.rho))

                        break
                    else:
                        continue
            
        
            act_combine =  self.action_space.from_vect(act.to_vect() + topo_act.to_vect())
            sim_obs, sim_reward, sim_done, sim_info = self.obs.simulate(act_combine)
            
            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                all_actions.append(copy.deepcopy(act_combine))
                all_rho.append(np.max(sim_obs.rho))


            act_id = np.argmin(all_rho)
            act    = all_actions[act_id]
                    
            new_obs, reward, is_done, _ = self.env.step(act)

            self.total_reward += reward
            new_state  = np.asarray(new_obs.to_vect()[chosen]) 
            
            # append transition when redispatching action is possible
            if act_id != 0:
                print("Act_id", act_id)
                print("\n")
                print(act)
                print("\n")
                exp = Experience(self.state, action, reward, is_done, new_state)
                self.exp_buffer.append(exp)
            
            self.state =  copy.deepcopy(new_state) 
            self.obs        =  copy.deepcopy(new_obs)
            
        else:
            act = copy.deepcopy(topo_act)
            new_obs, reward, is_done, _ = self.env.step(act)
            new_state  = np.asarray(new_obs.to_vect()[chosen]) 
            self.state =  copy.deepcopy(new_state) 
            self.obs        =  copy.deepcopy(new_obs)


        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward, self.obs

def calc_loss(batch, net, tgt_net, device="cpu"):
    
    states, actions, rewards, dones, next_states = batch
    states_v      = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v     = torch.tensor(actions).to(device)
    
    rewards_v     = torch.tensor(rewards).to(device)
    done_mask     = torch.BoolTensor(dones).to(device)
    
    actions_v     = actions_v.type(torch.int64)    
    
    state_action_values          = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values            = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values            = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)



if __name__ == "__main__":
    
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("device", device)
    env       = make("l2rpn_neurips_2020_track1_small")
    
    # observation shape 
    chosen    =  list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
    chosen    += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
    chosen    += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
    chosen    =  np.asarray(chosen, dtype=np.int32) - 1 
    obs       =  env.reset()
    # start sampling
    state     = np.asarray(obs.to_vect()[chosen])  
    in_dim    = len(state)
    
    action    = np.load('redispatch_act_40_actions.pickle', allow_pickle = True)
    act_dim   = len(action)
    
    # action shape
    
    net       = dqn_model.DQN(in_dim, act_dim ).to(device)
    tgt_net   = dqn_model.DQN(in_dim, act_dim).to(device)
    
    print(net)

    buffer    = ExperienceBuffer(REPLAY_SIZE)
    agent     = Agent(env, action, buffer)
    epsilon   = EPSILON_START

    optimizer        = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards    = []
    frame_idx        = 0
    ts_frame         = 0
    ts               = time.time()
    best_mean_reward = None

    epsilon          = 1.0
    rewards, lengths, losses, epsilons, dones = [], [], [], [], []
    
    N_EPISODES       = 2 # 10000 transition in every episode
    total_steps      = 10000
    
    for i in range(N_EPISODES):
        ep_reward, ep_loss = [], []
        total_rewards      = []
        frame_idx          = 0
        agent._reset()
        best_m_reward      = None
        done               = False
        step               = 0
        
        while step <= total_steps:
            epsilon          = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward , new_obs = agent.play_step(net, epsilon, device=device)
            obs              = copy.deepcopy(new_obs) 

            if reward is not None:
                print("done", done)
                done       = True
                total_rewards.append(reward)
                agent._reset()
                m_reward   = np.mean(total_rewards[-100:])
                print("%d: done %d games, reward %.3f, " "eps %.2f, step %.2f " % (
                    frame_idx, len(total_rewards), m_reward, epsilon,step))
            
            step           += 1
            
            if len(buffer) < REPLAY_START_SIZE:
                continue

            # soft copying the target from the main network
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
            
            optimizer.zero_grad()

            # sample the batch and performs the calculates the gradient  
            batch  = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            print("loss_t", loss_t)
            loss_t.backward()
            nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
=================================================================================================================



import grid2op
from grid2op import make
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

env = make("l2rpn_neurips_2020_track1_small")
obs = env.reset()

class DQN(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(DQN, self).__init__()
        
        n_cell = 1000
        self.layer1 = nn.Linear(in_dim, n_cell)
        self.layer2 = nn.Linear(n_cell,n_cell)
        self.layer3 = nn.Linear(n_cell,n_cell)
        self.layer4 = nn.Linear(n_cell,out_dim)
        
    def forward(self,obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        
        return x
    