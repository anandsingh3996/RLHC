"""
In this file, a PPO model is reproduced pytorch version of following OPENAI baselines' PPO model.
@https://github.com/openai/baselines/tree/master/baselines/ppo2
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import sys

class PPO(nn.Module): 
    def __init__(self, coef_entropy=0.01, coef_value_func=0.5, max_grad_norm=0.5):
        super(PPO, self).__init__()
        # this_directory_path = './'
        # act_shape = len(np.load(os.path.join(this_directory_path, 'topology_act_first_.pickle'), allow_pickle = True))
        self.model = Policy_Value_Network(1221,466)
        # print("act_shape",act_shape)
        self.optimizer =  optim.Adam(self.model.Model.parameters()) 
        self.coef_entropy = coef_entropy
        self.coef_value_func = coef_value_func
        self.max_grad_norm = max_grad_norm
        self.step = self.model.step
        self.value = self.model.value
        self.initial_state = None
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approx_kl', 'clip_ratio']


    def train(self, obs, returns, actions, values, neg_log_p_old, advs, lr = 3e-4, clip_range = 0.2):
        
        l, p, _ = self.model.Model(obs)

        #create one hot encoding of actions of shape of the logits 
        actions_one_hot = F.one_hot(actions.long(), num_classes=list(l.shape)[-1]).type('torch.FloatTensor') 
        
        # calculating the log probability of the logits 
        neg_log_p = F.cross_entropy(torch.unsqueeze(l, 0), torch.unsqueeze(actions_one_hot, 0)) 
        
        # calculate entropy bonus
        entropy = torch.mean(self._get_entropy(l)) # tf.reduce_mean(self._get_entropy(l))

        # calculate value loss
        vpred = self.model.value(obs)                                             # model value 
        vpred_clip = values +torch.clip(vpred - values, -clip_range, clip_range)  
        value_loss1 = torch.square(vpred - returns)                               # calculating the squared Error
        value_loss2 = torch.square(vpred_clip - returns)                          # calculating the squared error 

        value_loss = 0.5 * torch.mean(torch.maximum(value_loss1, value_loss2))# tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

        # calculate policy loss
        # Because the advantage is negative, the objective will increase if the action becomes less likely
        # that is, if \pi_{\theta}(a|s) decreases. But the max in this term puts a limit to how much the objective can increase. 
        # Once \pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s), the max kicks in and this term hits a ceiling 
        # of (1-\epsilon) A^{\pi_{\theta_k}}(s,a). Thus, again: the new policy does not benefit by going far away from the old policy.


        ratio = torch.exp(neg_log_p_old - neg_log_p) #torch.exp
        policy_loss1 = -advs * ratio
        policy_loss2 = -advs * torch.clip(ratio, (1 - clip_range), (1 + clip_range)) # clippiing the policy update
        policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))

        approx_kl = 0.5 * torch.mean(torch.square(neg_log_p_old - neg_log_p))
        clip_ratio = torch.mean((torch.greater(torch.abs(ratio - 1), clip_range)).type(torch.float))

        # Sigma loss
        loss = policy_loss * 10 + value_loss * self.coef_value_func - entropy * self.coef_entropy

        # model optimizer , get grad function uptill now, in train gradient losses are back propogated
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return policy_loss, value_loss, entropy, approx_kl, clip_ratio

    #save the model                
    def save(self, dir_path,update_num):
        torch.save(self.model.Model.state_dict(), os.path.join(dir_path, 'epoch-{}.pt'.format(update_num)))

    def load(self, dir_path):
        self.model.Model.load_state_dict(torch.load(os.path.join(dir_path,
                                         'ppo-ckpt/epoch_ckpt.pt')))

    #     self.model.Model.eval()

    def _get_entropy(self,l):
        a0 = l - torch.max(l)
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0)
        p0 = ea0/z0
        return torch.sum(p0* (torch.log(z0) - a0))


class Policy_Value_Network(object):
    
    def __init__(self, in_dim, out_dim):    
        # warm start from Junior Student
        self.Model = PVNet(in_dim, out_dim)

    def step(self, obs):
        # l for logits, p for possibility, and v for value
        l, p, v = self.Model(obs)

        # sampling by Gumbel-max trick
        u  = (torch.rand(l.shape)) 
        a = torch.argmax(l - torch.log(-torch.log(u)), axis = -1) 
        a_one_hot = (F.one_hot(a,num_classes=list(l.shape)[-1])).type('torch.FloatTensor')

        # calculate -log(pi)
        neg_log_p = F.cross_entropy(torch.unsqueeze(l, 0), torch.unsqueeze(a_one_hot, 0))
        neg_log_p = neg_log_p.unsqueeze(0)
        v = torch.squeeze(v, axis = 1)

        return a, v, neg_log_p, l
    
    def value(self, obs):
        _, _, v = self.Model(obs)
        v = torch.squeeze(v, axis=1)
        return v


class PVNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PVNet, self).__init__()
        n_cell = 1000

        self.layer1 = nn.Linear(in_dim, n_cell)
        self.layer2 = nn.Linear(n_cell,n_cell)
        self.layer3 = nn.Linear(n_cell,n_cell)
        self.layer4 = nn.Linear(n_cell,n_cell)
        self.act_layer = nn.Linear(n_cell, out_dim)
        self.val_hidden_layer = nn.Linear(n_cell, 64)
        self.val_layer = nn.Linear(64,1)


    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        s = F.relu(self.layer1(obs))
        s = F.relu(self.layer2(s))
        s = F.relu(self.layer3(s))
        s = F.relu(self.layer4(s))
        l = self.act_layer(s)                         # logits
        p = F.softmax(l, dim = 1)                     # probability distribution of actions
        vh = F.relu(self.val_hidden_layer(s))         # (1000-> 64)
        v = self.val_layer(vh)                        # state value(64-> 1)
        return l, p, v

