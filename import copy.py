import copy
import numpy as np


def line_disconnections(self, observation):      
    rho = copy.deepcopy(observation.rho)
    overflow = copy.deepcopy(observation.timestep_overflow)
    
    # 1) deals with soft overflow
    to_disc = (rho >=1.0) & (overflow == 3)
    # 2) disconnect lines on hard overflow
    to_disc[rho >2.0] = True

    if np.any(to_disc):
        tested_actions = []
        origin_rho = np.max(observation.rho)
        best_action = self.action_space({})
        line_id = None
        # line status change action
        for id_ in np.where(to_disc)[0]:
            change_status = self.action_space.get_change_line_status_vect()
            change_status[id_] = True
            action = self.action_space({"change_line_status": change_status})  
            sim_obs, sim_reward, sim_done, sim_info = observation.simulate(action)
            act_rho = np.max(sim_obs.rho)
            if not sim_info['is_illegal'] and not sim_info['is_ambiguous']:
                if not sim_done:
                    if act_rho < origin_rho:
                        best_action = copy.deepcopy(act_rho)
                        origin_rho = act_rho
                        line_id = id_
            return best_action,line_id
    return None , None  