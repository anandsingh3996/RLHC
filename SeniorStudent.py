"""
In this file, a multi-process training for PPO model is designed.

training process:
    The environment steps “do nothing” action (except reconnection of lines) until encountering a dangerous scenario, 
    then its observation is sent to the PPO model to get a “do something” action. After stepping this action, the reward is
    calculated and fed back to the PPO model for network updating.
"""
import os
import sys
import time
import grid2op
import numpy as np
from PPO import PPO
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PPO_Reward import PPO_Reward
from multiprocessing import cpu_count
from grid2op.Environment import SingleEnvMultiProcess


class Run_env(object):
    def __init__(self, envs, agent, n_steps, n_cores, gamma, lam, action_space_path='../'):
        #/content/drive/My Drive/l2rpn/ActionSpace
        self.envs = envs
        self.agent = agent
        self.n_steps = n_steps
        self.n_cores = n_cores
        self.gamma = gamma
        self.lam = lam

        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)

        # self.actions62 = np.load(os.path.join(action_space_path, 'actions62.npy'))
        # self.actions146 = np.load(os.path.join(action_space_path, 'actions146.npy'))
        self.actions = np.load(os.path.join(action_space_path, 'topology_act_first_.pickle'), allow_pickle = True) #(np.concatenate((self.actions62, self.actions146), axis=0)

        self.batch_reward_records = []
        self.aspace = self.envs.action_space[0]
        self.rec_rewards = []
        self.worker_alive_steps = np.zeros(n_cores)
        self.alive_steps_record = []

    def run_n_steps(self, n_steps=None):
        
        
        def swap_and_flatten(arr):
            shape = arr.shape
            output = arr.swapaxes(0, 1)
            return output.reshape(shape[0] * shape[1], *shape[2:])

        self.n_steps = n_steps if n_steps is not None else self.n_steps
        mb_obs, mb_rewards, mb_actions = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]
        mb_values, mb_dones, mb_neg_log_p = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]

        # start sampling
        obs_objs = self.envs.get_obs()
        obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])  # (12, 1221,)
        dones = np.asarray([False for _ in range(NUM_CORE)])  # (12,)
        agent_step_rs = np.asarray([0 for _ in range(NUM_CORE)], dtype=np.float64)  # (12,)

        for _ in range(self.n_steps):
            self.worker_alive_steps += 1
            actions = np.asarray([None for _ in range(NUM_CORE)])  
            values = np.asarray([None for _ in range(NUM_CORE)])
            neg_log_ps = np.asarray([None for _ in range(NUM_CORE)])
            for id in range(NUM_CORE):
                if obss[id, 654:713].max() >= ACTION_THRESHOLD:
                    actions[id], values[id], neg_log_ps[id], _ = map(lambda x: x.detach().numpy(), self.agent.step(torch.FloatTensor(obss[[id], :])))
                    if dones[id] == False and len(mb_obs[id]) > 0:
                        mb_rewards[id].append(agent_step_rs[id])
                    agent_step_rs[id] = 0
                    mb_obs[id].append(obss[[id], :])
                    mb_dones[id].append(dones[[id]])
                    dones[id] = False
                    mb_actions[id].append(actions[id])
                    mb_values[id].append(values[id])
                    mb_neg_log_p[id].append(neg_log_ps[id])
                else:
                    pass
            actions_array = []


            actions_array = [self.array2action(obs_objs[idx], self.actions[i][0], self.reconnect_array(obs_objs[idx])) if i is not None else self.array2action(obs_objs[idx], np.zeros(494), self.reconnect_array(obs_objs[idx])) for idx, i in enumerate(actions)]

            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)
            obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])
            for id in range(NUM_CORE):
                if env_dones[id]:
                    # death or end
                    self.alive_steps_record.append(self.worker_alive_steps[id])
                    self.worker_alive_steps[id] = 0
                    if 'GAME OVER' in str(infos[id]['exception']):
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] - 300)  
                    else:
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] + 500)
            agent_step_rs += rs
        # end sampling

        # batch to trajectory
        for id in range(NUM_CORE):
            if mb_obs[id] == []:
                continue
            if dones[id]:
                mb_dones[id].append(np.asarray([True]))
                mb_values[id].append(np.asarray([0]))
            else:
                mb_obs[id].pop()
                mb_actions[id].pop()
                mb_neg_log_p[id].pop()

        obs2ret, done2ret, action2ret, value2ret, neglogp2ret, return2ret = ([] for _ in range(6))
        for id in range(NUM_CORE):
            if mb_obs[id] == []:
                continue
            mb_obs_i = np.asarray(mb_obs[id], dtype=np.float32)
            mb_rewards_i = np.asarray(mb_rewards[id], dtype=np.float32)
            mb_actions_i = np.asarray(mb_actions[id], dtype=np.float32)
            mb_values_i = np.asarray(mb_values[id][:-1], dtype=np.float32)
            mb_neg_log_p_i = np.asarray(mb_neg_log_p[id], dtype=np.float32)
            mb_dones_i = np.asarray(mb_dones[id][:-1], dtype= bool)
            last_done = mb_dones[id][-1][0]
            last_value = mb_values[id][-1][0]

            # calculate R and A
            mb_advs_i = np.zeros_like(mb_values_i)
            last_gae_lam = 0
            for t in range(len(mb_obs[id]))[::-1]:
                if t == len(mb_obs[id]) - 1:
                    # last step
                    next_non_terminal = 1 - last_done
                    next_value = last_value
                else:
                    next_non_terminal = 1 - mb_dones_i[t + 1]
                    next_value = mb_values_i[t + 1]
                delta = mb_rewards_i[t] + self.gamma * next_value * next_non_terminal - mb_values_i[t]
                mb_advs_i[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            mb_returns_i = mb_advs_i + mb_values_i
            obs2ret.append(mb_obs_i)
            action2ret.append(mb_actions_i)
            value2ret.append(mb_values_i)
            done2ret.append(mb_dones_i)
            neglogp2ret.append(mb_neg_log_p_i)
            return2ret.append(mb_returns_i)

        obs2ret = np.concatenate(obs2ret, axis=0)
        action2ret = np.concatenate(action2ret, axis=0)
        value2ret = np.concatenate(value2ret, axis=0)
        done2ret = np.concatenate(done2ret, axis=0)
        neglogp2ret = np.concatenate(neglogp2ret, axis=0)
        return2ret = np.concatenate(return2ret, axis=0)
        self.rec_rewards.append(sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])

        list_el = list(map(swap_and_flatten, (obs2ret, return2ret, done2ret, action2ret, value2ret, neglogp2ret)))

        # print(list_el)
        obs_ = list(list_el[0])
        return_ = list(list_el[1])
        done_ = list(list_el[2])
        action_ = list(list_el[3])
        value_ = list(list_el[4])
        neg_log_p_ = list(list_el[5])

        rew_to_go  = (sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])
        
        return np.asarray(obs_,  dtype=np.float32), np.asarray(return_, dtype=np.float32), np.array(done_), np.asarray(action_, dtype=np.int32),np.asarray(value_, dtype=np.float32), np.asarray(neg_log_p_, dtype=np.float32), rew_to_go

    def reconnect_array(self, obs):

        reconnect_act_flag = False
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status == False)[0]

        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                line_to_reconnect = line  # reconnection
                new_line_status_array[line_to_reconnect] = 1
                reconnect_act_flag = True
                break
        return new_line_status_array.astype(int)
    
    # def array2action(self, obs, total_array, reconnect_array=None):
    #     action = self.aspace.from_vect(total_array)                     #({'change_bus': total_array[236:413]})
    #     action._change_bus_vect = action._change_bus_vect.astype(bool)
    #     if reconnect_array is None:
    #         return action
    #     action.update({'set_line_status': reconnect_array})
    #     return action
    def line_sub_impacted(self, action):
        topology_action_on_lines = []
        topology_action_on_subid = None 

        # topology_action_on_subid = action.impact_on_objects()['topology']['bus_switch'][0]['substation']

        for i in range(len(action.impact_on_objects()['topology']['bus_switch'])):
            object_type = action.impact_on_objects()['topology']['bus_switch'][i]['object_type']

            if object_type == 'line (extremity)' or object_type == 'line (origin)':
                object_id = action.impact_on_objects()['topology']['bus_switch'][i]['object_id']
                topology_action_on_lines.append(object_id)

        return topology_action_on_lines   #, topology_action_on_subid

        
    def array2action(self, obs, total_array, reconnect_array):
        
        disconnected_lines = np.where(obs.line_status == False)[0]  # to check disconnected lines information
        disc_line_cooldown_zero= []

        for i in disconnected_lines:
            if obs.time_before_cooldown_line[i] == 0:
                disc_line_cooldown_zero.append(i)

        reconnect_line = np.where(reconnect_array > 0)[0]           # check reconnect lines 
        action = self.aspace.from_vect(total_array)                 #({'change_bus': total_array[236:413]}) , encoding action
        act_lines = self.line_sub_impacted(action)         # check the lines and substations by the topology action 

        if len(np.where(reconnect_array == 0)[0]) == 59:            # reconnect_array is None:
            return action

        elif len(disc_line_cooldown_zero):                          # if there is any disconnected line present whose cooldown is zero
            if np.any(act_lines == reconnect_line):                 # reconnect_line is subset of it
                return action                                       # no need to combine reconnect action, it is aread included in the topology action")
            for i in disc_line_cooldown_zero:
                if np.any(act_lines == i):                          # disconnect line id with zero cooldown present in the action topo
                    return action                         
        else:                            
            action.update({'set_line_status': reconnect_array})     # if the reconnect line id is not not the lines affected by topology action then combine it 
            return action

        return action
    # def array2action(self, total_array, reconnect_array):

    #     # encoding action
    #     action = self.aspace.from_vect(total_array) #({'change_bus': total_array[236:413]})
        
    #     #reconnect_array is None:
    #     if len(np.where(reconnect_array == 0)[0]) == 59:    #reconnect_array is None:
    #         print("final_act",action)
    #         return action
    #     else:

    #         # if reconnect action is present, we need to check whether the action is ambiguous or not 

    #         line_affected_sub = []
    #         reconnect_line = np.where(reconnect_array == 1)[0]

    #         #reconnect line origin and extremety substation id 
    #         for i in reconnect_line:
    #             line_affected_sub.append(obs.line_or_to_subid[i])
    #             line_affected_sub.append(obs.line_ex_to_subid[i])

    #         #topology action affected substation 
    #         topology_action_sub_id = action.impact_on_objects()['topology']['assigned_bus'][0]['substation']

    #         # if reconnect line id's any substation(origin or extremety) is same as the topology action tageted substation
    #         # then take only topology action, no need to combine action else it will raise an error

    #         if np.any(line_affected_sub == topology_action_sub_id):
    #             return action
    #         else:
    #             # if the reconnect line id's substation is different from the topology targeted substation then combine bith actions 
                
    #             action.update({'set_line_status': reconnect_array})
    #             print("reconnect_array", reconnect_array)
    #             print("final_act",action)
        
    #     return action

if __name__ == '__main__':
    # hyper-parameters
   
    ACTION_THRESHOLD = 0.9
    DATA_PATH = '../l2rpn_neurips_2020_track1_small'         #/'content/drive/My Drive/l2rpn/training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../l2rpn_neurips_2020_track1_small/chronics'
    dir_path = '../ckpt'
    EPOCHS = 250
    NUM_ENV_STEPS_EACH_EPOCH = 10000
    logfile = open('../log/log_file_V.log','w')
    old_stdout = sys.stdout
    sys.stdout = logfile
    NUM_CORE = cpu_count()
    print('CPU counts：%d' % NUM_CORE)

    # Build single-process environment
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend, reward_class=PPO_Reward)
        print("lightsim2gridbackend")
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, reward_class=PPO_Reward)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    print("chronics reshuffed")
    # Convert to multi-process environment
    envs = SingleEnvMultiProcess(env=env, nb_env=NUM_CORE)
    # print("single environment multi process")
    envs.reset()
    print("mutiple environment reset")

    # Build PPO agent
    agent = PPO(coef_entropy=1e-3, coef_value_func=0.01, max_grad_norm=0.5, action_space_path='../ActionSpace')
    print("agent build ppo")

    # Build a runner
    print("going in runner")
    runner = Run_env(envs, agent, EPOCHS, NUM_CORE, gamma=0.99, lam=0.95,action_space_path='../ActionSpace')

    print("out of runner start training")
    
    for update in tqdm(range(EPOCHS)):
    # for update in range(EPOCHS):
        # update learning rate
        lr_now = 6e-5 * np.linspace(1, 0.025, 500)[np.clip(update, 0, 499)]
        if update < 5:
            lr_now = 1e-4
        clip_range_now = 0.2

        # generate a new batch
        tick = time.time()
        obs, returns, dones, actions, values, neg_log_p, ave_r = runner.run_n_steps(NUM_ENV_STEPS_EACH_EPOCH)
        print("runner steps done now in the trainig")
        print("returns", returns, len(returns))
        returns = returns/20
        print('sampling number in this epoch: %d' % obs.shape[0])
        # print("return", returns)

        # update policy-value-network
        
        n = obs.shape[0]
        advs = returns - values
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        # print("n and advantages", n, advs)
        sys.stdout.flush()

        for _ in range(2):
            ind = np.arange(n)
            np.random.shuffle(ind)
            for batch_id in range(10):
                slices = (torch.tensor(arr[ind[batch_id::10]]) for arr in (obs, returns, actions, values, neg_log_p, advs))
                policy_loss, value_loss, entropy, approx_kl, clip_ratio = agent.train(*slices,
                                                                                      lr=lr_now,
                                                                                      clip_range=clip_range_now)

        # logging
        print('epoch-%d, policy loss: %5.3f, value loss: %.5f, entropy: %.5f, approximate kl-divergence: %.5g, clipped ratio: %.5g' % (update, policy_loss, value_loss, entropy, approx_kl, clip_ratio))
        print('epoch-%d, ave_r: %5.3f, ave_alive: %5.3f, duration: %5.3f' % (update, ave_r, np.average(runner.alive_steps_record[-1000:]), time.time() - tick))
        sys.stdout.flush()

        # with open(logfile, 'a') as f:
        #     f.writelines('%d, %.2f, %.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.2f\n' % (update, ave_r, np.average(runner.alive_steps_record[-1000:]), policy_loss, value_loss, entropy, approx_kl, clip_ratio, time.time() - tick))

        #saving the model 
        agent.save(dir_path,update)
        print("done_training ")

    # sys.stdout = old_stdout
    # sys.stdout.flush()
    # logfile.flush()
    # logfile.close()
