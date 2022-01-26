from locale import normalize
import sys
import os, copy
import numpy as np
import pickle
from grid2op.Agent import BaseAgent

from .utils import utils, OUNoise, expBuffer # For Submission
from .my_model import DoubleDQN  # For Submission
from .my_model import ReplayBuffer           # For Submission
from .my_model import lineDisconnectionModel # For Submission

# from utils import utils, OUNoise, expBuffer  # For Training
# from my_model import DoubleDQN         # For Training
# from my_model import ReplayBuffer            # For Training
# from my_model import lineDisconnectionModel  # For Training

class MyAgent(BaseAgent):
    def __init__(self, env, dir_path):
        BaseAgent.__init__(self, env.action_space)  # action space converter

        self.obs_space = env.observation_space
        self.action_space = env.action_space

        # For Topology Agent
        self.nn               = 10
        self.batch_size       = 32
        self.rho_threshold    = 1.0
        self.action_size      = 5
        self.num_comb_thresh  = 150
        self.topo_action_used = 0

        normalization_values = pickle.load(open(os.path.join(dir_path,'Inputs/normalisation_constants.pickle'), "rb"))
        self.df = pickle.load(open(os.path.join(dir_path, 'Inputs/topological_actions_.pickle'), "rb"))
        self.sub_topo = pickle.load(open(os.path.join(dir_path,'Inputs/sub_topo.pickle'), "rb"))

        #heuristic_actions_file
        self.topo_action_sub_16 = pickle.load(open(os.path.join(dir_path, 'Inputs/sub_16_topo_vect'),"rb"))
        self.topo_action_sub_28 = pickle.load(open(os.path.join(dir_path, 'Inputs/sub_28_actions'),"rb"))
        self.topo_action_sub_23 = pickle.load(open(os.path.join(dir_path, 'Inputs/sub_23_actions'),"rb"))

        # it's used for calling state class
        self.utils = utils(env, normalization_values)

        # state_dim 1039 
        state_dim = (env.observation_space.n_gen *3 +
                     env.observation_space.n_line * 11 +
                     env.observation_space.n_load *3 +
                     env.observation_space.n_sub +
                     env.observation_space.dim_topo) # 1039

        # replay  buffer for dqn and experience replay for line disconnection
        self.replay_buffer_dqn  = ReplayBuffer()
        self.expBuffer = expBuffer()

        # For Disconnection Agent
        self.allowed_overflow_timestep = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED
        self.num_lines = len(self.action_space.line_or_to_subid)
        self.gen_pmax  = self.action_space.gen_pmax.max()
        self.lineModel = lineDisconnectionModel(7, self.allowed_overflow_timestep, self.rho_threshold)

        # For dqn Agent
        action_dim = len(self.df)
        self.dqn_action_space =  len(self.df)

        # Topology Model
        self.topoModel = DoubleDQN(state_dim, action_dim, self.df, self.utils, self.sub_topo)
    
    #save dqn and line disconnection Model
    def save(self):
        self.topoModel.save()
        self.lineModel.save()

    #load dqn and line disconnection Model
    def load(self, dir_path):
        self.topoModel.load(dir_path)
        self.lineModel.load()

    # reset experience buffer after epsiode is over
    def reset(self, obs):
        self.lineModel.reset()
        
# reset topology
# combine with topology
# reconnect
#  legal topo vect


    def act(self, observation, reward=None, done=False):

        # Fallback Action - Do Nothing
        do_nothing_action = self.action_space({})
        act = do_nothing_action

        all_actions = []
        sim_rho = []
        sim_rewards = []
        all_rew_acts = []
        non_ol_acts = []
        non_ol_reward = []
        
        disconnect_act ,disconnect_action = None, None
        reconnect_act , reconnect_action = None, None
        recovery_action = None
        topo_dqn_act = None
        disco_topo_act = None
        reco_topo_act = None
        
        present_rho =  np.max(observation.rho)

        self.line_disconnected = np.where(observation.line_status == False)[0].tolist()

        # simulating do nothing action first    
        do_nothing_action = self.action_space({})
        sim_obs,sim_reward,_,_ = observation.simulate(do_nothing_action)

        do_nothing_reward = sim_reward
        present_rho_threshold = np.max(sim_obs.rho)

        non_ol_acts.append(do_nothing_action)
        non_ol_reward.append(sim_reward)

        # do nothing will always be appended, 
        # try after removing it actually it makes the length of rho always > 0

        if present_rho_threshold <= present_rho:
            sim_rho.append(present_rho_threshold)
            all_actions.append(do_nothing_action)

        # Reconnect Action 
        act, line_id = self._reconnect_action(observation)

        # if act is not None:
        #     reconnect_act = line_id
        #     reconnect_action = act
        #     sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
        #     act_rho = np.max(sim_obs.rho)
            
        #     if not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
        #         if not sim_done:
        #             if  act_rho <= present_rho_threshold:
        #                 sim_rho.append(act_rho)
        #                 all_actions.append(act)

        #             if sim_reward >= do_nothing_reward:
        #                 all_rew_acts.append(act)
        #                 sim_rewards.append(sim_reward)

        #             non_ol_acts.append(act)
        #             non_ol_reward.append(sim_reward)

        if act is not None:
            reconnect_act = line_id
            reconnect_action = act

            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
            act_rho = np.max(sim_obs.rho)
            
            sim_rho.append(act_rho)
            all_actions.append(act)

            all_rew_acts.append(act)
            sim_rewards.append(sim_reward)

            non_ol_acts.append(act)
            non_ol_reward.append(sim_reward)

        # recover actions if the observation is not None 
        if observation is not None and not any(np.isnan(observation.rho)):
            if np.all(observation.topo_vect != -1):
                act = self._reset_topology(observation)
                recovery_action = act
                if recovery_action is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    act_rho = np.max(sim_obs.rho)

                    if not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if act_rho <= present_rho_threshold:
                                sim_rho.append(act_rho)
                                all_actions.append(act)

                            if sim_reward >= do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)

                            non_ol_acts.append(act)
                            non_ol_reward.append(sim_reward)

        # check for the overload 
        ol_list = self.getRankedOverloadList(observation)
        
        if len(ol_list) :
            # Line Disconnection Action  
            line_id = self.lineModel.select_Action(observation, self.num_lines, self.gen_pmax)

            if line_id is not None:
                act = self.action_space({"set_line_status": [(line_id,-1)]})
                disconnect_act = line_id
                disconnect_action = act
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                
                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:
                        act_rho = np.max(sim_obs.rho)
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)

                        if sim_reward >= do_nothing_reward:
                            sim_rewards.append(sim_reward)
                            all_rew_acts.append(act)

            # Topological Action  and combined with line actions
            state = self.utils.convert_obs(observation)
        
            #DQN actions             
            act, _, topo, sub_id = self.topoModel.select_action(state, observation, self.action_space)
                
            if (act != None and observation.time_before_cooldown_sub[sub_id] == 0):
                topo_legal =  self.legal_topo_vect(observation,sub_id, topo) 
                act = self.action_space({"set_bus":{"substations_id":[(sub_id, topo_legal)]}})  
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:
                        act_rho = np.max(sim_obs.rho)

                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)

                        if sim_reward >= do_nothing_reward:
                            sim_rewards.append(sim_reward)
                            all_rew_acts.append(act)
                
                # combining the dqn action with the line action 
                
                # if reconnect_act is not None:
                #     if (observation.line_or_to_subid[reconnect_act] != sub_id  and observation.line_ex_to_subid[reconnect_act] != sub_id):
                #         act = self.combine_with_topology(observation, reconnect_act, sub_id, topo_legal, 1)
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                #         reco_topo_act = act
                #         act_rho = np.max(sim_obs.rho)

                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)   

                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)
                            
                # if disconnect_act is not None:
                #     if  observation.line_or_to_subid[disconnect_act] != sub_id and observation.line_ex_to_subid[disconnect_act] != sub_id:
                #         act = self.combine_with_topology(observation, disconnect_act, sub_id,topo_legal, -1)
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                #         disco_topo_act = act
                #         act_rho = np.max(sim_obs.rho)

                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)

                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)

            act , act_vect = self.sub_16_act(observation,self.topo_action_sub_16 ,present_rho_threshold,do_nothing_reward)
            
            if act is not None:
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                sub_id_ = 16
                topo_vector = act_vect
                
                act_rho = np.max(sim_obs.rho)

                if act_rho <= present_rho_threshold:
                    sim_rho.append(act_rho)
                    all_actions.append(act)

                if sim_reward >= do_nothing_reward:
                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)


                # if reconnect_action is not None:
                #     if (observation.line_or_to_subid[reconnect_act] != 16 and observation.line_ex_to_subid[reconnect_act] != 16):
                        
                #         act_ = self.legal_topo_vect(observation, sub_id_, topo_vector)   
                #         act = self.combine_with_topology(observation, reconnect_act, sub_id_,
                #                                      act_, 1)                        
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                #         self.reco_topo_act = act

                #         act_rho = np.max(sim_obs.rho)
                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)
                                                                    
                # if disconnect_act is not None:
                #     if  observation.line_or_to_subid[disconnect_act] != 16 and observation.line_ex_to_subid[disconnect_act] != 16:
                #         act_  = self.legal_topo_vect(observation, sub_id_, topo_vector)  
                #         act = self.combine_with_topology(observation, disconnect_act, sub_id_,
                #                                      act_, -1)
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                #         disco_topo_act = act
                        
                #         act_rho = np.max(sim_obs.rho)    

                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)

                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                                    
            
            act, act_vect = self.sub_28_act(observation,self.topo_action_sub_28 ,present_rho_threshold,do_nothing_reward)
            
            if act is not None:
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                sub_id_ = 28
                topo_vector = act_vect
                act_rho = np.max(sim_obs.rho)
                if act_rho <= present_rho_threshold:
                    sim_rho.append(act_rho)
                    all_actions.append(act)

                if sim_reward >= do_nothing_reward:
                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)

                # if reconnect_action is not None:
                #     if (observation.line_or_to_subid[reconnect_act] != 28 and observation.line_ex_to_subid[reconnect_act] != 28):
                        
                #         act_ = self.legal_topo_vect(observation, sub_id_, topo_vector)   
                #         act = self.combine_with_topology(observation, reconnect_act, sub_id_,
                #                                      act_, 1)                        
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                #         self.reco_topo_act = act

                #         act_rho = np.max(sim_obs.rho)
                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)
                                                                    
                # if disconnect_act is not None:
                #     if  observation.line_or_to_subid[disconnect_act] != 28 and observation.line_ex_to_subid[disconnect_act] != 28:
                #         act_  = self.legal_topo_vect(observation, sub_id_, topo_vector)  
                #         act = self.combine_with_topology(observation, disconnect_act, sub_id_,
                #                                      act_, -1)
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                #         disco_topo_act = act
                        
                #         act_rho = np.max(sim_obs.rho)    

                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)

                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                                    

            act , act_vect = self.sub_23_act(observation,self.topo_action_sub_23 ,present_rho_threshold,do_nothing_reward)
            
            if act is not None:
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                sub_id_ = 23
                topo_vector = act_vect

                act_rho = np.max(sim_obs.rho)
                if act_rho <= present_rho_threshold:
                    sim_rho.append(act_rho)
                    all_actions.append(act)

                if sim_reward >= do_nothing_reward:
                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    

                # if reconnect_action is not None:
                #     if (observation.line_or_to_subid[reconnect_act] != 23 and observation.line_ex_to_subid[reconnect_act] != 23):
                        
                #         act_ = self.legal_topo_vect(observation, sub_id_, topo_vector)   
                #         act = self.combine_with_topology(observation, reconnect_act, sub_id_,
                #                                      act_, 1)                        
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                #         self.reco_topo_act = act

                #         act_rho = np.max(sim_obs.rho)
                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)
                                                                    
                # if disconnect_act is not None:
                #     if  observation.line_or_to_subid[disconnect_act] != 23 and observation.line_ex_to_subid[disconnect_act] != 23:
                #         act_  = self.legal_topo_vect(observation, sub_id_, topo_vector)  
                #         act = self.combine_with_topology(observation, disconnect_act, sub_id_,
                #                                      act_, -1)
                #         sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                #         disco_topo_act = act
                        
                #         act_rho = np.max(sim_obs.rho)    

                #         if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                #             if not sim_done:
                #                 if sim_reward >= do_nothing_reward:
                #                     sim_rewards.append(sim_reward)
                #                     all_rew_acts.append(act)

                #                 if act_rho <= present_rho_threshold:
                #                     sim_rho.append(act_rho)
                #                     all_actions.append(act)
                                    
            

            if len(sim_rho) == 0:
                if len(self.line_disconnected) > 0:
                    if len(sim_rewards) > 0:
                        best_act_idx_reward = np.argmax(sim_rewards)
                        act_reward = all_rew_acts[best_act_idx_reward]
                        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act_reward)
                        act = act_reward
                    else:
                        act = do_nothing_action
                else:
                    act = do_nothing_action

            elif len(sim_rho) > 0:
                best_act_idx_rho = np.argmin(sim_rho)
                act_rho = all_actions[best_act_idx_rho]
                sim_obs,sim_rho,sim_done,sim_info = observation.simulate(act_rho)
                act = act_rho
        
        else:
            if len(sim_rewards) > 0:
                best_act_idx_reward = np.argmax(sim_rewards)
                act_reward = all_rew_acts[best_act_idx_reward]
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act_reward)

                act = act_reward

            else:
                act = do_nothing_action
        return act


    def sub_23_act(self,observation,sub_23_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub23 = []
        sim_rho_sub23 = []
        legal_sub_23_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 23
        topo_vect = None

        if observation.time_before_cooldown_sub[substation_id_] == 0:
            for i in range(len(sub_23_action)):
                topo_legal = self.legal_topo_vect(observation, substation_id_,sub_23_action[i][1] )
                action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    sim_rewards_sub23.append(sim_reward)
                    sim_rho_sub23.append(np.max(sim_obs.rho))
                    legal_sub_23_vect.append(topo_legal)

            if len(sim_rho_sub23) > 0:
                best_act_rho = np.argmin(sim_rho_sub23)
                new_rho = sim_rho_sub23[best_act_rho]
                
                if new_rho <= present_rho_threshold:
                    topo_rho_legal_vect = legal_sub_23_vect[best_act_rho]
                    act = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_rho_legal_vect)]}})
                    best_action = act
                    topo_vect = topo_rho_legal_vect
            else:
                best_action = None
                topo_vect = None
        
        return best_action ,topo_vect

    def sub_28_act(self,observation,sub_28_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub28 = []
        sim_rho_sub28 = []
        legal_sub_28_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 28
        topo_vect = None

        if observation.time_before_cooldown_sub[substation_id_] == 0:
            for i in range(len(sub_28_action)):
                topo_legal = self.legal_topo_vect(observation, substation_id_,sub_28_action[i][1] )
                action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    sim_rewards_sub28.append(sim_reward)
                    sim_rho_sub28.append(np.max(sim_obs.rho))
                    legal_sub_28_vect.append(topo_legal)

            if len(sim_rho_sub28) > 0:
                best_act_rho = np.argmin(sim_rho_sub28)
                new_rho = sim_rho_sub28[best_act_rho]
                
                if new_rho <= present_rho_threshold:
                    topo_rho_legal_vect = legal_sub_28_vect[best_act_rho]
                    rho_act = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_rho_legal_vect)]}})

                    best_action = rho_act
                    topo_vect = topo_rho_legal_vect
            else:
                best_action = None
                topo_vect = None

        return best_action, topo_vect
        
    def sub_16_act(self,observation,sub_16_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub16 = []
        sim_rho_sub16 = []
        legal_sub_16_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 16
        topo_vect = None

        if observation.time_before_cooldown_sub[substation_id_] == 0:
            for i in range(len(sub_16_action)):
                topo_legal = self.legal_topo_vect(observation, substation_id_,sub_16_action[i] )
                action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    sim_rewards_sub16.append(sim_reward)
                    sim_rho_sub16.append(np.max(sim_obs.rho))
                    legal_sub_16_vect.append(topo_legal)

        if len(sim_rho_sub16)> 0:
            best_act_rho = np.argmin(sim_rho_sub16)
            new_rho = sim_rho_sub16[best_act_rho]

            if new_rho <= present_rho_threshold:
                topo_rho_legal_vect = legal_sub_16_vect[best_act_rho]
                rho_act = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_rho_legal_vect)]}})
                topo_vect = topo_rho_legal_vect
                best_action = rho_act

        else:
            best_action = None
            topo_vect = None

        return best_action, topo_vect

    def getAction(self, observation, epsilon,train=True):    # modification in act function before test

        # Fallback Action - Do Nothing
        action_for_buffer = {}
        do_nothing_action = self.action_space({})
        act = do_nothing_action

        rew_counter  = -1
        rho_counter  = -1

        all_rho_acts = []
        all_rew_acts = []

        sim_rewards   = []
        sim_rho       = []

        disconnect_act  = None
        reconnect_act   = None
        recovery_action = None

        rew_action_flag = {}
        rho_action_flag = {}

        features = []
        dqn_topo_buffer   = None

        self.line_disconnected = np.where(observation.line_status == False)[0].tolist()

        do_nothing_action = self.action_space({})
        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(do_nothing_action)
        do_nothing_reward = sim_reward

        present_rho = np.max(sim_obs.rho)

        sim_rewards.append(sim_reward)
        all_rew_acts.append(do_nothing_action)
        rew_counter += 1
        rew_action_flag[rew_counter] = 'do_nothing'

        sim_rho.append(np.max(sim_obs.rho))
        all_rho_acts.append(do_nothing_action)
        rho_counter += 1
        rho_action_flag[rho_counter] = 'do_nothing'

        # Reconnect Action
        act, line_id = self._reconnect_action(observation)
        reconnect_act = line_id

        if act is not None:
            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

            if sim_reward >= do_nothing_reward:
                sim_rewards.append(sim_reward)
                all_rew_acts.append(act)
                rew_counter += 1
                rew_action_flag[rew_counter] = 'reconnect'

            if np.max(sim_obs.rho) <= present_rho:
                sim_rho.append(np.max(sim_obs.rho))
                all_rho_acts.append(act)
                rho_counter += 1
                rho_action_flag[rho_counter] = 'reconnect'

        #recover actions if the observation is not None
        if observation is not None and not any(np.isnan(observation.rho)):
            if np.all(observation.topo_vect != -1):

                act = self._reset_topology(observation)
                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                    if sim_reward >= do_nothing_reward:
                        sim_rewards.append(sim_reward)
                        all_rew_acts.append(act)
                        rew_counter += 1
                        rew_action_flag[rew_counter] = 'recovery'

                    if np.max(sim_obs.rho) <= present_rho:
                        sim_rho.append(np.max(sim_obs.rho))
                        all_rho_acts.append(act)
                        rho_counter += 1
                        rho_action_flag[rho_counter] = 'recovery'

        ol_list = self.getRankedOverloadList(observation)

        if len(ol_list) or len(self.line_disconnected) > 0:
            # Line Disconnection Action
            line_id = self.lineModel.select_Action(observation, self.num_lines,
                                                   self.gen_pmax, epsilon, train)

            if line_id is not None:
                act = self.action_space({"set_line_status": [(line_id,-1)]})
                disconnect_act = line_id
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                if sim_reward >= do_nothing_reward:
                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'disconnect'

                if np.max(sim_obs.rho) <= present_rho:
                    sim_rho.append(np.max(sim_obs.rho))
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'disconnect'

            # Topological Action  and combined with line actions
            state = self.utils.convert_obs(observation)

            # -------------------------------- Topo DQN
            if train:
                if np.random.random() <= epsilon:
                    act, dqn_topo_buffer, topo, sub_id = self.exploring_actions(observation)
                else:
                    act, dqn_topo_buffer, topo, sub_id = self.topoModel.select_action(state,
                                                                                      observation,
                                                                                      self.action_space)
            else:
                act, dqn_topo_buffer, topo, sub_id = self.topoModel.select_action(state,
                                                                                  observation,
                                                                                  self.action_space)

            if (act != None and observation.time_before_cooldown_sub[sub_id] == 0):

                topo_legal = self.legal_topo_vect(observation, sub_id, topo)
                act = self.action_space({"set_bus":{"substations_id":[(sub_id, topo_legal)]}})

                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:
                        if sim_reward >= do_nothing_reward:
                            sim_rewards.append(sim_reward)
                            all_rew_acts.append(act)
                            rew_counter += 1
                            rew_action_flag[rew_counter] = 'topo_dqn'

                        if np.max(sim_obs.rho) <= present_rho:
                            sim_rho.append(np.max(sim_obs.rho))
                            all_rho_acts.append(act)
                            rho_counter += 1
                            rho_action_flag[rho_counter] = 'topo_dqn'

                # combining the dqn action with the line action

                if reconnect_act is not None:
                    act = self.combine_with_topology(observation, reconnect_act, sub_id,
                                                     topo_legal, 1)
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if sim_reward >= do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'reconnect_topo_dqn'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'reconnect_topo_dqn'


                if disconnect_act is not None:
                    act = self.combine_with_topology(observation, disconnect_act, sub_id,
                                                     topo_legal, -1)
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if sim_reward >= do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'disconnect_topo_dqn'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'disconnect_topo_dqn'

            # Explore DQN for sub 16 (sub ids with > threshold topo combos)

            sub_id_ = 16
            if observation.time_before_cooldown_sub[sub_id_] == 0:
                act, act_vect = self.sub_16_act(observation, self.topo_action_sub_16,
                                      present_rho, do_nothing_reward)

                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'topo_sub16'


                    sim_rho.append(sim_reward)
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'topo_sub16'

            sub_id_ = 23
            if observation.time_before_cooldown_sub[sub_id_] == 0:
                act, act_vect = self.sub_23_act(observation, self.topo_action_sub_23,
                                      present_rho, do_nothing_reward)

                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'topo_sub23'


                    sim_rho.append(sim_reward)
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'topo_sub23'

            sub_id_ = 28
            if observation.time_before_cooldown_sub[sub_id_] == 0:
                act, act_vect = self.sub_28_act(observation, self.topo_action_sub_28,
                                      present_rho, do_nothing_reward)

                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'topo_sub28'


                    sim_rho.append(sim_reward)
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'topo_sub28'

            #adding one more condition to check if line is disconnected then take the action based on the reward

            if len(sim_rho) == 1 and len(self.line_disconnected) > 0:
                if len(sim_rewards) > 1:
                    best_act_idx_reward = np.argmax(sim_rewards)
                    act_reward = all_rew_acts[best_act_idx_reward]
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act_reward)

                    act = act_reward
                    best_act_idx = best_act_idx_reward
                    action_flag = rew_action_flag

                else:
                    act = do_nothing_reward
                    best_act_idx = None
                    action_flag = None


            else:
                if len(sim_rho) > 0:
                    best_act_idx_rho = np.argmin(sim_rho)
                    act_rho = all_rho_acts[best_act_idx_rho]
                    sim_obs,sim_rho,sim_done,sim_info = observation.simulate(act_rho)

                    act = act_rho
                    best_act_idx = best_act_idx_rho
                    action_flag = rho_action_flag
        
        else:
            if len(sim_rewards) > 1:
                best_act_idx_reward = np.argmax(sim_rewards)
                act_reward = all_rew_acts[best_act_idx_reward]
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act_reward)

                act = act_reward
                best_act_idx = best_act_idx_reward
                action_flag = rew_action_flag

            else:
                act = do_nothing_reward
                best_act_idx = None
                action_flag = None


        if 'disconnect' in action_flag[best_act_idx]:
            action_for_buffer['disconnect'] = disconnect_act
        if 'topo_dqn' in action_flag[best_act_idx]:
            action_for_buffer['topo_dqn']   = dqn_topo_buffer
        if 'reconnect' in action_flag[best_act_idx]:
            action_for_buffer['reconnect']  = None
        if 'topo_sub16' in action_flag[best_act_idx]:
            action_for_buffer['topo_sub16'] = None
        if 'topo_sub23' in action_flag[best_act_idx]:
            action_for_buffer['topo_sub23'] = None
        if 'topo_sub28' in action_flag[best_act_idx]:
            action_for_buffer['topo_sub28'] = None
        if 'do_nothing' in action_flag[best_act_idx]:
            action_for_buffer['do_nothing'] = None
        if 'recovery' in action_flag[best_act_idx]:
            action_for_buffer['recovery'] = None

        return act, action_for_buffer, features

    def exploring_actions(self, observation):
        counter  = 0
        act_flag = {}
        random_actions = np.random.choice(np.arange(self.dqn_action_space), 
                                          self.dqn_action_space, 
                                          replace=False)
        for indx, q_indx in enumerate(random_actions):
            topo_vect,enc_act = self.Q_vals_action_encoding(observation,q_indx, self.action_space)
            
            best_action,best_action_id = self.select_best_action_from_dual(observation, enc_act)
            topo_vect_ = topo_vect[0][1][best_action_id]
            substation_topo = topo_vect[0][0]
            
            _,sim_done,sim_done,sim_info = observation.simulate(best_action)
            
            if not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
                topo_act = best_action
                act_flag[counter] = 'topology'
                counter += 1   
                act_id = q_indx 
                return topo_act, act_id,topo_vect_,substation_topo
        
        return None, None, None, None

    def Q_vals_action_encoding(self,obs, Q, a_space):
        sub_id = []
       
        actions_topology = []
        
        acts = []
                        
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
            # if sub_id != 64:
            for l in range(len(actions_topology[j][1])):
                act = a_space({"set_bus": {"substations_id": [(sub_id, actions_topology[j][1][l])]}})
                sub_act.append(act)
                sub_ID.append([sub_id, actions_topology[j][1][l]])
                
            # else:
            #     sub_act.append(a_space({}))

        return actions_topology,sub_act

    def _reset_topology(self, observation):
        if np.max(observation.rho) <= 0.95: #check if max rho less than 0.95
            for sub_id, sub_elem_num in enumerate(observation.sub_info): #enumate subid and sub element
                sub_topo = observation.state_of(substation_id=sub_id)["topo_vect"]   #self.sub_toop_dict[sub_id] #getting topology of given sub id from dict

                if sub_id == 28:
                    sub28_topo = np.array([2, 1, 2, 1, 1]) #
                    if not np.all(
                            sub_topo.astype(int) == sub28_topo.astype(int)
                    ) and observation.time_before_cooldown_sub[
                            sub_id] == 0:
                        sub_id = 28

                        act = self.action_space({
                            "set_bus": {
                                "substations_id": [(sub_id, sub28_topo)]
                            }
                        })

                        obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                            act)
                        # observation._obs_env._reset_to_orig_state()
                        assert not info_simulate[
                            'is_illegal'] and not info_simulate['is_ambiguous']
                        if not done_simulate and obs_simulate is not None and not any(
                                np.isnan(obs_simulate.rho)):
                            if np.max(obs_simulate.rho) <= 0.95:
                                return act
                    continue



                if np.any(
                        sub_topo == 2
                ) and observation.time_before_cooldown_sub[sub_id] == 0:
                    sub_topo = np.where(sub_topo == 2, 1,
                                        sub_topo)  # bus 2 to bus 1
                    sub_topo = np.where(sub_topo == -1, 0,
                                        sub_topo)  # don't do action in bus=-1
                    reconfig_sub = self.action_space({
                        "set_bus": {
                            "substations_id": [(sub_id, sub_topo)]
                        }
                    })

                    obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                        reconfig_sub)
                    # observation._obs_env._reset_to_orig_state()

                    assert not info_simulate[
                        'is_illegal'] and not info_simulate['is_ambiguous']

                    if not done_simulate:
                        assert np.any(
                            obs_simulate.topo_vect !=
                            observation.topo_vect)  # have some impact

                    if not done_simulate and obs_simulate is not None and not any(
                            np.isnan(obs_simulate.rho)):
                        if np.max(obs_simulate.rho) <= 0.95:
                            return reconfig_sub


        if np.max(observation.rho) >= 1.0:
            sub_id = 28
            sub_topo =  observation.state_of(substation_id=sub_id)["topo_vect"] # self.sub_toop_dict[sub_id]
            if np.any(
                    sub_topo == 2
            ) and observation.time_before_cooldown_sub[sub_id] == 0:
                sub28_topo = np.array([1, 1, 1, 1, 1])
                act = self.action_space({
                    "set_bus": {
                        "substations_id": [(sub_id, sub28_topo)]
                    }
                })
                obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                    act)
                # observation._obs_env._reset_to_orig_state()
                assert not info_simulate['is_illegal'] and not info_simulate[
                    'is_ambiguous']
                if not done_simulate and obs_simulate is not None and not any(
                        np.isnan(obs_simulate.rho)):
                    if np.max(obs_simulate.rho) <= 0.99:
                        return act



    def combine_with_topology(self, observation, line_id, sub_id, topo_act, line_act):
        new_line_status_array = np.zeros(observation.rho.shape).astype(int)
        new_line_status_array[line_id] = line_act

        line_el_idx = -1
        # check if line is in substation
        # If the sub is the line origin
        if sub_id == observation.line_or_to_subid[line_id]:
            line_el_idx = observation.line_or_to_sub_pos[line_id]

        # If the sub is the line extrimity
        elif sub_id == observation.line_ex_to_subid[line_id]:
            line_el_idx = observation.line_ex_to_sub_pos[line_id]

        if line_el_idx > -1:
            topo_act[line_el_idx] = 0

        action_space_ = {}
        action_space_["set_bus"] = {}

        action_space_["set_bus"]["substations_id"] =  [(sub_id, topo_act)]
        action_space_["set_line_status"] = new_line_status_array

        act_combined = self.action_space(action_space_)

        return act_combined

    def legal_topo_vect(self,observation, sub_id, topo_vect_):

        topo_vect = observation.state_of(substation_id=sub_id)["topo_vect"]

        do_nothing_indices = np.where(topo_vect <= 0)[0]
        if len(do_nothing_indices)>0:
            for i in do_nothing_indices:
                topo_vect_[i] = 0
        return topo_vect_

    def _reconnect_action(self, observation):
        disconnected = np.where(observation.line_status == False)[0].tolist()
        action = None
        line_id = None
        
        for line_id in disconnected:
            if observation.time_before_cooldown_line[line_id] == 0:
                action = self.action_space({"set_line_status": [(line_id, +1)]})
                obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(action)
                if np.max(observation.rho) <= 1.0 and np.max(obs_simulate.rho) >= 1.0:
                    continue

                line_id = line_id

        return action,line_id

    def getRankedOverloadList(self, observation):
        sort_rho = -np.sort(-observation.rho)  # sort in descending order for positive values
        sort_indices = np.argsort(-observation.rho)
        line_list = [sort_indices[i] for i in range(len(sort_rho))
                     if sort_rho[i] >= self.rho_threshold]
        return line_list


    def train_dqn(self, current_step_num, batch_size, discount, policy_freq):
        if batch_size <= self.replay_buffer_dqn.getLength():
            return self.topoModel.train(self.replay_buffer_dqn,
                                   current_step_num,
                                   batch_size, discount,
                                   policy_freq)

    def update_PolicyNet(self):
        return self.lineModel.updatePolicy()

def make_agent(env, this_directory_path):
    # Add l2rpn reward
    my_agent = MyAgent(env, this_directory_path)
    my_agent.load(this_directory_path)
    return my_agent
