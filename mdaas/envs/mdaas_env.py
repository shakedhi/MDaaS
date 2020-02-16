from gym import spaces, Env
import pandas as pd
import numpy as np
import itertools


class MdaasEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 features_path, 
                 proba_costs_path, 
                 cost_func=lambda r: r ** .5, 
                 t_func=lambda r: 1, 
                 fp_func=lambda r: -r, 
                 fn_func=lambda r: -r,
                 illegal_reward=-10000,
                 failed_scan_cost=1):
        
        # init action space
        self.termination_actions_ = ['benign', 'malicious']
        self.scan_actions_ = ['pefile', 'byte3g', 'opcode2g', 'cuckoo']
        self.multi_actions_ = []
        for i in range(1, len(self.scan_actions_) + 1):
            self.multi_actions_ += [set(x) for x in itertools.combinations(self.scan_actions_ ,i)]
        
        self.action_labels = self.multi_actions_ + self.termination_actions_
        self.action_space = spaces.Discrete(n=len(self.action_labels))

        # init state
        num_features = [0] * 4
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(sum(num_features) + len(num_features), 1))
        self.term_state_ = np.full(shape=(self.observation_space.shape[0],), fill_value=-2, dtype=np.float32)
        self.state_index_, prev = [], 0
        for t in num_features:
            self.state_index_.append((prev, prev + t))
            prev += t

        # init costs
        self.costs = dict(manalyze=0.75, pefile=0.7, byte3g=3.99, opcode2g=42.99, cuckoo=138.6, failed=failed_scan_cost)
        
        # init data
        self.proba_costs = pd.read_csv(proba_costs_path).sort_values(by=['Name']).reset_index(drop=True)
        self.features = pd.read_csv(features_path).sort_values(by=['Name']).reset_index(drop=True)
        self.num_files = self.proba_costs.shape[0]
        
        # init reward functions
        self.illegal_reward = illegal_reward
        self.cost_func = cost_func
        self.t_func = t_func
        self.fp_func = fp_func
        self.fn_func = fn_func
        
        # reset
        self.reset_epoch()
        self.reset()
    
    def step_illegal_(self, pred):
        self.done = True
        self.reward = self.illegal_reward
        self.pred = pred
        self.state = self.term_state_
        return self.state, self.reward, self.done, {}

    def step_scan_(self, action_labels):
        cost = 0
        for action_label in action_labels:
            action = self.scan_actions_.index(action_label)
            start, end = self.state_index_[action]
            features = self.current['features'][start:end]
            result = self.current['proba_costs']['Proba_' + action_label]
            
            cost_t = self.costs[action_label] if result >= 0 else self.costs['failed']
                
            cost = max(cost, cost_t)

            # update state
            start, end = start + action, end + action + 1
            self.state[start:end] = features + [result]
        
        # update reward
        self.reward += self.cost_func(cost)
        self.total_time += cost
        
        return self.state, 0, self.done, {}
    
    def step_term_(self, action_label):
        if self.current['label'] == 1:
            # Actual malicious
            if action_label is 'malicious':
                self.reward = self.t_func(self.reward)
                self.pred = 'TP'
            else:
                self.reward = self.fn_func(self.reward)
                self.pred = 'FN'
        else:
            # Actual benign
            if action_label is 'benign':
                self.reward = self.t_func(self.reward)
                self.pred = 'TN'
            else:
                self.reward = self.fp_func(self.reward)
                self.pred = 'FP'

        self.done = True
        return self.state, self.reward, self.done, {}
    
    def step(self, action):
        action_label = self.action_labels[action]
        self.scanned.append(action_label)
        
        if action_label in self.multi_actions_:
            unique_actions = set().union(*self.scanned[:-1])
            if unique_actions.intersection(action_label) != set():
                return self.step_illegal_(pred='DUP')  # Duplicate action
            return self.step_scan_(action_label)
        
        elif action_label in self.termination_actions_:
            if len(self.scanned) == 1:
                return self.step_illegal_(pred='DIR')  # Direct classification
            return self.step_term_(action_label)
        
        raise ValueError('action value is outside of action_space')
    
    def reset(self):
        i, pc = next(self.proba_costs_iter, (None, None))
        f = self.features[self.features['Name'] == pc['Name']].iloc[0]
        
        self.current = {
            'name': f['Name'],
            'label': f['Label'],
            'features': f.values.tolist(),
            'proba_costs': pc
        }

        self.state = np.full(shape=(self.observation_space.shape[0],), fill_value=-1, dtype=np.float32)
        self.reward = 0
        self.done = False
        self.scanned = []
        self.pred = ''
            
        if i == (self.num_files - 1):
            self.epoch_end = True
        
        return self.state
            
    def reset_epoch(self):
        self.epoch_end = False
        self.total_time = 0
        self.proba_costs_iter = self.proba_costs.iterrows()

    def render(self, mode='human', close=False):
        if self.done:
            l = 'm' if self.current['label'] == 1 else 'b'
            n = self.current['name'][:8]
            a = ','.join(self.scanned)
            print('[%s-%s] R=%d\t%s\tA=%s' % (l, n, self.reward, self.pred, a))
