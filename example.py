from chainer.initializers import LeCunNormal
from chainer.optimizer import GradientClipping
from chainer import links as L, functions as F

from chainerrl.optimizers.rmsprop_async import RMSpropAsync
from chainerrl.replay_buffer import EpisodicReplayBuffer
from chainerrl.distribution import SoftmaxDistribution
from chainerrl.action_value import DiscreteActionValue
from chainerrl.links import Sequence
from chainerrl.agents.acer import ACERSeparateModel
from chainerrl.agents import ACER
from chainerrl import misc

from datetime import datetime
from detector import Detector
import numpy as np
import random
import mdaas
import time
import gym


class MockTimedDetector(Detector):
    def __init__(self):
        pass

    def detect(self, fd):
        start = datetime.now()
        time.sleep(random.uniform(0.5, 2))
        result = random.uniform(0, 1)
        end = datetime.now()
        duration = (end - start).total_seconds()
        return result, duration


def create_agent(train=True):
    model = ACERSeparateModel(
        pi=Sequence(
            L.Linear(env.observation_space.shape[0], 20),
            F.relu,
            L.Linear(20, env.action_space.n, initialW=LeCunNormal(1e-3)), 
            SoftmaxDistribution,
        ),
        q=Sequence(
            L.Linear(env.observation_space.shape[0], 20),
            F.relu,
            L.Linear(20, env.action_space.n, initialW=LeCunNormal(1e-3)), 
            DiscreteActionValue,
        ) 
    )

    opt = RMSpropAsync(lr=7e-4, eps=1e-2, alpha=0.99)
    opt.setup(model)
    opt.add_hook(GradientClipping(40))

    agent = ACER(
        model=model,                               # Model to train
        optimizer=opt,                             # The optimizer
        gamma=0.99,                                # Reward discount factor
        t_max=50,                                  # The model is updated after this many local steps
        replay_buffer=EpisodicReplayBuffer(5000),  # The replay buffer
        replay_start_size=1000,                    # Replay buffer won't be used until it has at least this many episodes
        beta=1,                                    # Entropy regularization parameter
    )

    if not train:
        agent.act_deterministically = True
    
    return agent


def process_file(file_data, train=True):
    reward, done = 0, False
    state = env.reset(file_data=file_data)
    while not done:
        if train:
            action = agent.act_and_train(state, reward=reward)
        else:
            action = agent.act(state)
        
        state, reward, done, _info = env.step(action)

    # End episode
    if train:
        agent.stop_episode_and_train(state, reward, done)        
    else:
        agent.stop_episode()

    # Dump model on train
    if train:
        agent.save('.')  
    
    return env.pred, env.costs


if __name__ == '__main__':
    misc.set_random_seed(123)

    detectors = {
        'pefile': MockTimedDetector(),
        'byte3g': MockTimedDetector(),
        'opcode2g': MockTimedDetector(),
        'cuckoo': MockTimedDetector()
    }

    env = gym.make(
        'MDaaS-v1',
        detectors=detectors,          # Action labels
        cost_func=lambda r: r ** .5,  # Cost function
        t_func=lambda r: 1,           # Reward for TP or TN
        fp_func=lambda r: -r,         # Reward for FP
        fn_func=lambda r: -r,         # Reward for FN
        illegal_reward=-10000         # Cost for illegal action
    )

    train = True

    agent = create_agent(train=train)

    # Mock files
    predictions, times = [], []
    for i in range(10):
        f = {
            'Name': str(i),
            'Label': round(random.uniform(0, 1)),
            'File': None  # File descriptor/Path
        }
        p, t = process_file(f, train=train)
        predictions.append(p)
        times.append(sum(t))
    
    true_rate = predictions.count('TP') + predictions.count('TN')
    accuracy = (100 * true_rate) / len(predictions)
    total_time = sum(times) / len(times)
    print(f'Reached an accuracy of {accuracy} in {total_time:.2f} seconds per file.')
