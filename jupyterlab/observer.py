import multigamepy
import paratroopergame
import math
# Import environment base class for a wrapper 
from gymnasium import Env 

# Import the space shapes for the environment
from gymnasium.spaces import Discrete, Box, MultiBinary
# Import numpy to calculate frame delta 
import numpy as np

from stable_baselines3 import MultiSB3, PPO, DQN

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import BaseCallback

import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

import pickle_skip


LOG_DIR = './logs/'
OPT_DIR = './models/'

MULTIGAME = multigamepy.MultiGameManager()


class Paratrooper(Env): 
    def __init__(self, render_mode = 'human'):
        super().__init__()
        
        # Startup and instance of the game 
        self.game = paratroopergame.GameSystem('Paratrooper game', paratroopergame.GAME_MODE_EXT_ACTION, render_mode)

        # Specify action space and observation space
        self.render_mode = render_mode
        self.observation_space = Box(low=0, high=255, shape=(45, 80, 1), dtype=np.uint8)
        self.action_space = MultiBinary(3)
    
    def reset(self, seed = 0):       
        super().reset(seed=seed)
        # Return the first frame 
        obs = self.game.reset(seed)
        obs = [obs]

        info = {}

        info['none'] = 0

        self.LastElapsedTime = 0
        self.LastDestroyedParatroopers = 0
        self.LastMissedBullets = 0
        self.LastEscapedParatroopers = 0
        
        return obs, info
    
    def step(self, action): 
        # Take a step 
        obs, done, trimmed, info = self.game.step(action)

        # There is only one observation element
        obs = [obs]

        reward = [0,0]

        reward[0] = info['ElapsedTime'] - self.LastElapsedTime
        reward[0] += (info['DestroyedParatroopers'] - self.LastDestroyedParatroopers) * 10

        if(info['LowestParatrooperExists']):
            targetVector = [0,-1]
        else:
            _lp = info['LowestParatrooperPosition']
            _cp = info['CannonPosition']
            _ccs = info['CannonAngleCosSin']
            targetVector = [_lp[0] - _cp[0], _lp[1] - _cp[0]]

            #If paratrooper is lower than cannon, its impossible to reach
            if(targetVector[1] > 0):
                targetVector[1] = 0

        _dotproduct = _ccs[0] * targetVector[0] + _ccs[1] * targetVector[1]
            
        reward[1] = _dotproduct


        self.LastElapsedTime = info['ElapsedTime']
        self.LastDestroyedParatroopers = info['DestroyedParatroopers']
        self.LastMissedBullets = info['MissedBullets']
        self.LastEscapedParatroopers = info['EscapedParatroopers']
        
        return obs, reward, done, trimmed, info
    
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        self.game.close()



# Define class for periodic stoarge of trained models
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, algo_name, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.algo_name = algo_name
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}_{}'.format(self.algo_name,self.n_calls))
            self.model.save(model_path)

        # It is necessary to return True. Otherwise learn process would end
        return True



# Create environment. AI tells game not to render or process window events, which makes FPS higher
real_env = Paratrooper('ai')
real_env = Monitor(real_env, LOG_DIR)

# Algorithm list with deferred actions would be next:
# 0: DQN (shot)
# 1: PPO (aim)
observation_list = []
action_space_list = []

action_space_list.append(Discrete(2)) # DQN (DON'T SHOT, Action SHOT), DQN can only cope with Discrete Spaces
action_space_list.append(MultiBinary(2)) # PPO (Action Right, Left)
observation_list.append(real_env.observation_space) # DQN will observe element 0 of obs return
observation_list.append(real_env.observation_space) # PPO will observe element 0 of obs return (Same as DQN)

# Every algorithm can take its own periodicity to save model
callback_0 = TrainAndLoggingCallback(check_freq=20000, save_path=OPT_DIR, algo_name='DQN')
callback_1 = TrainAndLoggingCallback(check_freq=20000, save_path=OPT_DIR, algo_name='PPO')

callback_list = [callback_0, callback_1]

# Create virtual environments
virtual_env_list = MultiSB3.createVirtualEnvironments(numAlgorithms=2, observationSpaceList=observation_list, actionSpaceList=action_space_list)

# Create algorithms, by association of its indexed virtual environment with them
alg_0 = DQN('CnnPolicy', virtual_env_list[0], tensorboard_log=LOG_DIR)
alg_1 = PPO('CnnPolicy', virtual_env_list[1], tensorboard_log=LOG_DIR)
#alg_0 = DQN.load(os.path.join(OPT_DIR, 'best_model_DQN_x.zip'), env=env)
#alg_1 = PPO.load(os.path.join(OPT_DIR, 'best_model_PPO_x.zip'), env=env)

# Dictionary for DQN will specify DQN model itself, and obs_index 0 to pick first element from real environment return
alg_collection_0 = {}
alg_collection_0['alg'] = alg_0
alg_collection_0['env'] = virtual_env_list[0]
alg_collection_0['obs_index'] = 0
alg_collection_0['reward_index'] = 0
alg_collection_0['action_indexes'] = [0]
alg_collection_0['action_space'] = action_space_list[0]

# Dictionary for PPO will specify PPO model itself, and obs_index 0 to pick first element from real environment return
alg_collection_1 = {}
alg_collection_1['alg'] = alg_1
alg_collection_1['env'] = virtual_env_list[1]
alg_collection_1['obs_index'] = 0
alg_collection_1['reward_index'] = 1
alg_collection_1['action_indexes'] = [1,2]
alg_collection_1['action_space'] = action_space_list[1]

alg_collection = [alg_collection_0, alg_collection_1]



# Create MultiAlgorithm, this one is in contact with real environment and deffers actions and observation to virtual environments
model = MultiSB3(real_env, alg_collection, virtual_env_list)

observable_inst = pickle_skip.PickleSkipper(model)
observable1 = observable_inst.Update()

model.learn(total_timesteps=10000, callback_alg=callback_list)
#env.close()