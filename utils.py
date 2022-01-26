import numpy as np

class utils():
    def __init__(self, env, normalization_values):       
        obs = env.reset()
        self.normalize_value = normalization_values

        # gen_normalise_values 
        self.p_gen_max = self.normalize_value['gen']['p_gen_max'] 
        self.p_gen_min = self.normalize_value['gen']['p_gen_min']
        self.p_gen_mean = self.normalize_value['gen']['p_gen_mean'] 
        self.p_gen_std = self.normalize_value['gen']['p_gen_std']

        self.q_gen_max = self.normalize_value['gen']['q_gen_max'] 
        self.q_gen_min = self.normalize_value['gen']['q_gen_min'] 
        self.q_gen_mean = self.normalize_value['gen']['q_gen_mean']
        self.q_gen_std = self.normalize_value['gen']['q_gen_std']

        self.v_gen_max = self.normalize_value['gen']['v_gen_max']
        self.v_gen_min = self.normalize_value['gen']['v_gen_min']
        self.v_gen_mean = self.normalize_value['gen']['v_gen_mean']
        self.v_gen_std = self.normalize_value['gen']['v_gen_std']

        #load_normalise_values
        self.p_load_max = self.normalize_value['load']['p_load_max'] 
        self.p_load_min = self.normalize_value['load']['p_load_min'] 
        self.p_load_mean = self.normalize_value['load']['p_load_mean'] 
        self.p_load_std = self.normalize_value['load']['p_load_std'] 

        self.q_load_max = self.normalize_value['load']['q_load_max']
        self.q_load_min = self.normalize_value['load']['q_load_min'] 
        self.q_load_mean = self.normalize_value['load']['q_load_mean']
        self.q_load_std = self.normalize_value['load']['q_load_std'] 

        self.v_load_max = self.normalize_value['load']['v_load_max'] 
        self.v_load_min = self.normalize_value['load']['v_load_min'] 
        self.v_load_mean = self.normalize_value['load']['v_load_mean']
        self.v_load_std = self.normalize_value['load']['v_load_std']

        #line normalise values 
        self.p_line_or_max = self.normalize_value['line']['p_or_max']
        self.p_line_or_min = self.normalize_value['line']['p_or_min']

        self.q_line_or_max = self.normalize_value['line']['q_or_max'] 
        self.q_line_or_min = self.normalize_value['line']['q_or_min']
        self.q_line_or_mean = self.normalize_value['line']['q_or_mean'] 
        self.q_line_or_std = self.normalize_value['line']['q_or_std'] 

        self.a_line_or_max = self.normalize_value['line']['a_or_max'] 
        self.a_line_or_min = self.normalize_value['line']['a_or_min'] 
        self.a_line_or_mean = self.normalize_value['line']['a_or_mean'] 
        self.a_line_or_std = self.normalize_value['line']['a_or_std'] 
            
        self.v_line_or_max = self.normalize_value['line']['v_or_max']
        self.v_line_or_min = self.normalize_value['line']['v_or_min'] 
        self.v_line_or_mean = self.normalize_value['line']['v_or_mean'] 
        self.v_line_or_std = self.normalize_value['line']['v_or_std']

        self.p_line_ex_max = self.normalize_value['line']['p_ex_max'] 
        self.p_line_ex_min = self.normalize_value['line']['p_ex_min'] 
 
        self.q_line_ex_max = self.normalize_value['line']['q_ex_max'] 
        self.q_line_ex_min = self.normalize_value['line']['q_ex_min'] 
        self.q_line_ex_mean =self.normalize_value['line']['q_ex_mean'] 
        self.q_line_ex_std = self.normalize_value['line']['q_ex_std']

        self.a_line_ex_max = self.normalize_value['line']['a_ex_max']
        self.a_line_ex_min = self.normalize_value['line']['a_ex_min']
        self.a_line_ex_mean = self.normalize_value['line']['a_ex_mean']
        self.a_line_ex_std = self.normalize_value['line']['a_ex_std']

        self.v_line_ex_max = self.normalize_value['line']['v_ex_max']
        self.v_line_ex_min = self.normalize_value['line']['v_ex_min'] 
        self.v_line_ex_mean = self.normalize_value['line']['v_ex_mean'] 
        self.v_line_ex_std = self.normalize_value['line']['v_ex_std'] 

        self.obs_space = env.observation_space
        self.act_space = env.action_space
        
    def convert_obs(self, observation):

        gen_features = [] # 66
        gen_features.append(((observation.prod_p - self.p_gen_min)/( self.p_gen_max -  self.p_gen_min)))
        gen_features.append(((observation.prod_q - self.q_gen_min)/( self.q_gen_max -  self.q_gen_min)))
        gen_features.append(((observation.prod_v - self.v_gen_min)/( self.v_gen_max -  self.v_gen_min)))
        gen_features = np.concatenate(gen_features)

        load_features = [] # 111
        load_features.append(((observation.load_p - self.p_load_min)/( self.p_load_max -  self.p_load_min)))
        load_features.append(((observation.load_q - self.q_load_min)/( self.q_load_max -  self.q_load_min)))
        load_features.append(((observation.load_v - self.v_load_min)/( self.v_load_max -  self.v_load_min)))
        load_features = np.concatenate(load_features)

        line_or_features = [] # 236
        line_or_features.append(((observation.p_or -  self.p_line_or_min)/( self.p_line_or_max -  self.p_line_or_min)))
        line_or_features.append(((observation.q_or -  self.q_line_or_min)/( self.q_line_or_max -  self.q_line_or_min)))
        line_or_features.append(((observation.v_or -  self.v_line_or_min)/( self.v_line_or_max -  self.v_line_or_min)))
        line_or_features.append(((observation.a_or -  self.a_line_or_min)/( self.a_line_or_max -  self.a_line_or_min)))
        line_or_features = np.concatenate(line_or_features)

        line_ex_features = [] # 236
        line_ex_features.append(((observation.p_ex - self.p_line_ex_min)/( self.p_line_ex_max -  self.p_line_ex_min)))
        line_ex_features.append(((observation.q_ex - self.q_line_ex_min)/( self.q_line_ex_max -  self.q_line_ex_min)))
        line_ex_features.append(((observation.v_ex - self.v_line_ex_min)/( self.v_line_ex_max -  self.v_line_ex_min)))
        line_ex_features.append(((observation.a_ex - self.a_line_ex_min)/( self.a_line_ex_max -  self.a_line_ex_min)))
        line_ex_features = np.concatenate(line_ex_features)

        rho = observation.rho - 1.0 # 59
        features = np.concatenate([load_features, gen_features, line_or_features, line_ex_features, rho])
                
        topo_vector = 1/ (observation.topo_vect + 1) # 177
        sub_cooldown_info = 1/ ( observation.time_before_cooldown_sub + 1) # 36
        line_cooldown_info = 1/ ( observation.time_before_cooldown_line + 1) # 59
        line_overflow_info = 1/ (observation.timestep_overflow + 1) # 59

        other_features = np.concatenate([topo_vector, sub_cooldown_info, line_cooldown_info, line_overflow_info])

        final_features = np.concatenate([features, other_features])  # (1039)

        return final_features.tolist() 

class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

#policy agent replay buffer

from collections import deque

class expBuffer():
    def __init__(self, buffer_size = 500):
        self.buffer = deque([])
        self.buffer_size = buffer_size
    
    def get_size(self):
        return len(self.buffer)
    
    def add(self,experience,cap=1):
        if len(self.buffer) + len(experience) >= self.buffer_size and cap:
            for _ in range(len(experience)):
                self.buffer.popleft()
        self.buffer.extend(experience)
    
    def clear(self):
        del self.buffer; self.buffer = deque([])
            
    def sample(self, sample_index):
        mySample = []
        buffer   = list(self.buffer)
        for _i in sample_index:
            mySample.append(buffer[_i])
        return mySample