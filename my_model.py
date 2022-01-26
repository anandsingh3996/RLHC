import numpy as np
import os
import copy
import sys

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .utils import expBuffer
# from utils import expBuffer


nn_device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(nn_device)

criterion = nn.MSELoss()


class ReplayBuffer(object):
    def __init__(self, max_size=10000):  
        self.max_size = max_size
        self.ptr = 0
        self.storage = []
        
    def getLength(self):
        return len(self.storage)

    def add(self, transition):
        # print("STORAGE",len(self.storage))
        if len(self.storage) == (self.max_size):
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % (self.max_size)
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        for i in ind:
            data = self.storage[i]
            state, action, reward, next_state, done = data
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_dones.append(np.array(done, copy=False))

        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_next_states = np.array(batch_next_states)
        batch_dones = np.array(batch_dones)

        return batch_states,  batch_actions, batch_rewards,\
            batch_next_states, batch_dones


class DDQN(nn.Module):      
    
    def __init__(self, observation_size, action_size):
        super(DDQN,self).__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        
        self.layer_1 = nn.Linear(observation_size,1024)
        self.layer_2 = nn.Linear(1024,512)
        self.layer_3 = nn.Linear(512,256)
        self.layer_4 = nn.Linear(256, action_size)   
        
    def forward(self,x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x 
    

class DoubleDQN(object):    
    def __init__(self, state_size, action_size, df, utils, sub_topo):

        self.Q_main = DDQN(state_size, action_size).to(nn_device)
        self.Q_target = DDQN(state_size, action_size).to(nn_device)
        self.Q_target.load_state_dict(self.Q_main.state_dict())
        self.Q_main_optimizer =  T.optim.SGD(self.Q_main.parameters(), lr=0.01, momentum=0.9) #T.optim.Adam(self.Q_main.parameters())
                
        self.state_size  = state_size
        self.action_size = action_size
        self.utils = utils
        self.df    = df
        self.sub_topo = sub_topo
        self.loss = T.nn.L1Loss()

    def select_action(self, state, observation, action_space):
        topo_act, topo_vect_, act_id , sub_id = None, None, None, None

        state  = T.FloatTensor(state).to(nn_device)
        q_actions  = self.Q_main(state).cpu().data.numpy().flatten()
        self.action_space = action_space

        q_vals = q_actions    
          
        sorted_idx = np.argsort(q_vals)[::-1]     

        for q_indx in sorted_idx:
            topo_vect,enc_act = self.Q_vals_action_encoding(observation,q_indx, 
                                                            self.action_space)
            
            best_action,best_action_id = self.select_best_action_from_dual(observation, 
                                                                           enc_act)
            topo_vect_ = topo_vect[0][1][best_action_id]
            sub_id = topo_vect[0][0] 
            _,_,_,sim_info = observation.simulate(best_action)
            
            if not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
                topo_act = best_action
                best_action,best_action_id = self.select_best_action_from_dual(observation, 
                                                                               enc_act)
                topo_vect_ = topo_vect[0][1][best_action_id]
                sub_id = topo_vect[0][0]
                act_id = q_indx 
                break 
        
        if str(nn_device) == 'cuda': T.cuda.empty_cache()

        return topo_act, act_id, topo_vect_, sub_id

    def select_best_action_from_dual(self, observation, encoded_action):
        best_action = None
        enc_act = encoded_action
        act_flag = {}
        counter  = 0
        a_reward = []
        a_rho = []
        
        for i in range(len(enc_act)):
            sim_obs,sim_reward,_,sim_info = observation.simulate(enc_act[i])
            a_reward.append(sim_reward)
            a_rho.append(np.max(sim_obs.rho))
            act_flag[counter] = 'enc_act_' + str(i)
            counter +=1

        # best_act_idx = np.argmax(a_reward)
        best_act_idx = np.argmin(a_rho)
        
        if act_flag[best_act_idx] == 'enc_act_0':
            best_action = enc_act[0]
            
        elif act_flag[best_act_idx] == 'enc_act_1':
            best_action = enc_act[1]
        
        return best_action,best_act_idx
    
    def Q_vals_action_encoding(self,obs,Q, a_space):
        sub_id = []
       
        actions_topology = []
        acts = []
        if Q == 226:
            actions_topology.append((226, self.df[Q]))       
        else:
            a = self.df[Q][0] #substation id 
            arr = copy.deepcopy(self.sub_topo[a])
            arr_2 = copy.deepcopy(self.sub_topo[a])
            
            for node in self.df[Q][1][0]:
                arr[node-1] = 1
                arr_2[node-1] = 2
            for node in self.df[Q][1][1]:
                arr[node-1] = 2
                arr_2[node-1] = 1
                
            acts.append(arr)
            acts.append(arr_2)
            actions_topology.append((a,acts))
            sub_id.append(a)
            
        sub_act = []
        sub_ID = []
        
        for j in range(len(actions_topology)):
            sub_id = actions_topology[j][0]
            if sub_id != 226: #226
                for l in range(len(actions_topology[j][1])):
                    act = a_space({"set_bus": {"substations_id": [(sub_id, actions_topology[j][1][l])]}})
                    sub_act.append(act)
                    sub_ID.append([sub_id, actions_topology[j][1][l]])           
            else:
                sub_act.append(a_space({}))

        return actions_topology,sub_act

    # for trainig the DQN network once total_timesteps > learning steps 
    def train(self, replay_buffer, total_timesteps, batch_size, discount , policy_freq):

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        state = T.FloatTensor(states).to(nn_device)
        next_state = T.FloatTensor(next_states).to(nn_device)
        action = T.tensor(actions).to(nn_device)
        reward = T.FloatTensor(rewards).to(nn_device)
        done = T.FloatTensor(dones).to(nn_device)

        print(state.shape, next_state.shape, action.shape, reward.shape, done.shape)
        action = action.unsqueeze(-1)
        print(action, action.shape)


        '''state_action_val = self.Q_main(state)
        state_action_val = state_action_val.gather(1,action.long())
        state_action_val = state_action_val.squeeze(-1)
        with no_grad(): #don't want gradients for model
            next_state_acts = self.Q_main(next_state).max(1)[1] #max(1)[1] gives us index of the largest value while max(1)[0] gives the direct value 
        next_state_acts = next_state_acts.unsqueeze(-1)

        """ then use these actions to get Q values from the target network """

        next_state_vals = self.Q_target(next_state).gather(1,next_state_acts).squeeze(-1)
        done = done.squeeze(-1)
        reward = (reward/1000) # divide by 1000 
        # print("BATCH REWARD MEAN",reward.mean())
        
        exp_sa_vals = next_state_vals.detach()*discount*(1-done) + reward

        Q_main_loss = F.mse_loss(state_action_val,exp_sa_vals)
        loss = Q_main_loss.detach().cpu().numpy()

        #back propagating the Q_main loss and update the parameters of the model using SGD optimizer
        self.Q_main_optimizer.zero_grad()
        Q_main_loss.backward()
        self.Q_main_optimizer.step()'''

        current_Q_values = self.Q_main(state)
        # We choose Q based on action taken
        current_Q_values = current_Q_values.gather(1, action).squeeze(-1)   

        next_state_vals = self.Q_target(next_state).max(1)[0].detach()   
        done = done.squeeze(-1)

        target_Q_values = reward + (next_state_vals * discount* (1-done))

        Q_main_loss =  self.loss(target_Q_values , current_Q_values)
        print("Q_main_loss", Q_main_loss)

        self.Q_main_optimizer.zero_grad()
        Q_main_loss.backward()
        self.Q_main_optimizer.step()
        
        #Here we copy the weights of the main network to the target network at some defined iterations
        if total_timesteps % (policy_freq) == 0:
            self.Q_target.load_state_dict(self.Q_main.state_dict())
        
        # if str(nn_device) == 'cuda': T.cuda.empty_cache()
            
        return Q_main_loss.detach().cpu().numpy()  #loss 
      
    #save the model
                
    def save(self):
        T.save(self.Q_main.state_dict(), 'MySavedModel/MyDQNModel.pth')

    #load the model      
    def load(self, dir_path):
        self.Q_main.load_state_dict(T.load(os.path.join(dir_path,
                                         'MySavedModel/MyDQNModel.pth'),
                                           map_location=nn_device))


class PolicyNet(nn.Module):
    def __init__(self, ip_num):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(ip_num, 16)
        self.fc2 = nn.Linear(16, 256)
        self.fc3 = nn.Linear(256, 4096)
        self.fc4 = nn.Linear(4096, 256)
        self.fc5 = nn.Linear(256, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        x = T.FloatTensor(x).to(nn_device)
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        x = T.tanh(self.fc4(x))
        x = T.tanh(self.fc5(x))
        x = T.tanh(self.fc6(x))
        x = T.sigmoid(x)
        return x


class lineDisconnectionModel():
    def __init__(self, num_ip, allowed_overflow_timestep, rho_threshold):
        self.model = PolicyNet(num_ip).to(nn_device)
        self.targt = PolicyNet(num_ip).to(nn_device)
        self.num_ip = num_ip
        
        self.gamma = 0.99
        self.bsize = 1000
        self.iterations  = 20
        self.optimizer   = T.optim.SGD(self.model.parameters(), lr=0.01)
        self.threshold_powerFlow_safe = rho_threshold # earlier 0.95 -> 35.95
        
        self.allowed_overflow_timestep = allowed_overflow_timestep        
        
        self.pstate_pool = expBuffer()
        self.nstate_pool = expBuffer()
        self.reward_pool = expBuffer()
        self.epdone_pool = expBuffer()
        
        self.add_reward  = 0
        
    def save(self):
        T.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
            }, 'MySavedModel/MyLineModel.pt')
        
    def reset(self):
        self.ep_state_pool = []
        self.ep_action_pool = []
        self.ep_reward_pool = []

    def load(self):
        path = 'MySavedModel/MyLineModel.pt'
        if os.path.exists(path):
            # print("Using Saved Weights", end=" ")
            checkpoint = T.load(path, map_location=nn_device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def addReward(self,reward):
        self.reward_pool.append(reward)
      
    def select_Action(self, observation, num_lines, 
                      gen_pmax, epsilon=0, train=False):
        line_mask = np.zeros(num_lines)
        features = []
        self.features = None
        
        #select action from the model 
        
        for line_id in range(num_lines):
            features.append([observation.p_or[line_id]/gen_pmax,
                              observation.q_or[line_id]/gen_pmax,
                              observation.p_ex[line_id]/gen_pmax,
                              observation.q_ex[line_id]/gen_pmax,
                              observation.v_or[line_id]/220,
                              observation.v_ex[line_id]/220,
                              observation.rho[line_id]])                    
            if observation.timestep_overflow[line_id] >=\
                self.allowed_overflow_timestep:
                line_mask[line_id] = 1
        
        if np.any(line_mask):
            features = np.reshape(features, (-1,7))
            q_vals = self.model(features).data.cpu().numpy().flatten()     
            q_vals = np.where(line_mask == 1, q_vals, -np.ones(num_lines))  
              
            if train and np.random.random() < epsilon:
                line_id = np.random.choice(num_lines)
            else:
                line_id = np.argmax(q_vals)
            
            self.features = features
            return line_id
        
        else:
            return None
        
    def addToBuffer(self, action, reward):
        self.ep_state_pool.append(self.features)
        self.ep_action_pool.append(action)
        self.ep_reward_pool.append(reward)       
    
    def updatePolicy(self):
        if len(self.ep_state_pool) != len(self.ep_action_pool) or\
           len(self.ep_state_pool) != len(self.ep_reward_pool) or\
           len(self.ep_action_pool)!= len(self.ep_reward_pool):
            print("Episodic Buffers not equal!! Exit!!")
            sys.exit()
        if len(self.ep_state_pool) == 0:
            print("Line Buffer is Zero!!")
            # sys.exit()
            
        pstates = []
        nstates = []
        rewards = []
        
        count = 0
        for step in range(len(self.ep_state_pool)):
            if self.ep_action_pool[step] != -1 and\
               step+1 < len(self.ep_state_pool):
                line_id = self.ep_action_pool[step]
                pstates.append(self.ep_state_pool[step][line_id])
                nstates.append(self.ep_state_pool[step+1][line_id])
                rewards.append(self.ep_reward_pool[step])
                count += 1
        
        del self.ep_action_pool
        del self.ep_reward_pool
        del self.ep_state_pool
        
        # Discount reward
        reward_pool = np.zeros(len(rewards))
        running_add = 0
        for i in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[i]
            reward_pool[i] = running_add
            
        del rewards
            
        self.pstate_pool.add(pstates); del pstates
        self.nstate_pool.add(nstates); del nstates
        self.reward_pool.add(reward_pool); del reward_pool
        
        if self.pstate_pool.get_size() != self.nstate_pool.get_size() or\
           self.pstate_pool.get_size() != self.reward_pool.get_size():
            print("Main Buffers not equal!! Exit!!")
            sys.exit()
        
        if self.pstate_pool.get_size() < 5:
            return 0

        losses = []
        criterion = nn.SmoothL1Loss()
        
        if self.bsize > self.pstate_pool.get_size():
            batch_size = self.pstate_pool.get_size()//2
        else:
            batch_size = self.bsize
        iterations = min(self.iterations, self.pstate_pool.get_size()//batch_size)

        for i in range(iterations):
            indices = np.random.choice(np.arange(self.pstate_pool.get_size()), size=batch_size, replace=False)

            pstates = self.pstate_pool.sample(indices)
            nstates = self.nstate_pool.sample(indices)
            rewards = self.reward_pool.sample(indices)
            
            
            # rewards = (rewards - np.mean(rewards)) / max(1,np.std(rewards))
            rewards = np.reshape(rewards, (-1,1))

            # nstates = np.reshape(nstates, (-1,14))
            nstates = np.reshape(nstates, (-1,7))
            tvalues = self.targt(nstates)

            # pstates = np.reshape(pstates, (-1,14))
            pstates = np.reshape(pstates, (-1,7))
            qvalues = self.model(pstates)
            
            rewards = T.FloatTensor(rewards).to(nn_device)
            target  = rewards + self.gamma*tvalues
            
            errors  = criterion(qvalues,target)

            losses.append(errors.detach().cpu().numpy())
            self.optimizer.zero_grad()
            errors.backward()
            self.optimizer.step()
            if str(nn_device) == 'cuda': T.cuda.empty_cache()
        
        return np.sum(losses)
