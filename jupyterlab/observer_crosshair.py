import multigamepy
import crosshairgame
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


class Crosshair(Env): 
    def __init__(self, render_mode = 'human'):
        super().__init__()
        
        # Startup and instance of the game 
        self.game = crosshairgame.GameSystem('Crosshair game', crosshairgame.GAME_MODE_EXT_ACTION, render_mode)

        # Specify action space and observation space
        self.render_mode = render_mode
        self.observation_space = Box(low=0, high=255, shape=(45, 80, 1), dtype=np.uint8)
        #self.observation_space = Box(low=0, high=255, shape=(90, 160, 1), dtype=np.uint8)
        self.action_space = MultiBinary(4)
    
    def reset(self, seed = 0):       
        super().reset(seed=seed)
        # Return the first frame 
        obs, info = self.game.reset(seed)
        obs = [obs]

        self.LastInsideTime = 0
        
        return obs, info
    
    def step(self, action): 
        # Take a step 
        obs, done, trimmed, info = self.game.step(action)

        # There is only one observation element
        obs = [obs]

        reward = [0,0]
        
        reward[0] = (64 - abs(info['DeltaPosition'][1]))/10.0
        reward[1] = (64 - abs(info['DeltaPosition'][0]))/10.0
        
        reward[0] += info['InsideTime'] - self.LastInsideTime
        reward[1] += info['InsideTime'] - self.LastInsideTime
        
        self.LastInsideTime = info['InsideTime']
        
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
real_env = Crosshair('human')
#real_env = Monitor(real_env, LOG_DIR)

# Algorithm list with deferred actions would be next:
# 0: DQN (shot)
# 1: PPO (aim)
observation_list = []
action_space_list = []

action_space_list.append(Discrete(3)) # DQN (Up, Down, don't move), DQN can only cope with Discrete Spaces
action_space_list.append(Discrete(3)) # PPO (Action Right, Left, don't move)
observation_list.append(real_env.observation_space) # DQN will observe element 0 of obs return
observation_list.append(real_env.observation_space) # PPO will observe element 0 of obs return (Same as DQN)

# Every algorithm can take its own periodicity to save model
callback_0 = TrainAndLoggingCallback(check_freq=20000, save_path=OPT_DIR, algo_name='DQN')
callback_1 = TrainAndLoggingCallback(check_freq=20000, save_path=OPT_DIR, algo_name='PPO')

callback_list = [callback_0, callback_1]

# Create virtual environments
virtual_env_list = MultiSB3.createVirtualEnvironments(numAlgorithms=2, observationSpaceList=observation_list, actionSpaceList=action_space_list)

# Use Monitor for them
virtual_env_list[0] = Monitor(virtual_env_list[0], LOG_DIR)
virtual_env_list[1] = Monitor(virtual_env_list[1], LOG_DIR)

# Create algorithms, by association of its indexed virtual environment with them
#alg_0 = PPO('MlpPolicy', virtual_env_list[0], tensorboard_log=LOG_DIR, n_steps=64)
#alg_1 = PPO('MlpPolicy', virtual_env_list[1], tensorboard_log=LOG_DIR, n_steps=64)
alg_0 = PPO.load(os.path.join(OPT_DIR, 'best_model_DQN_200000.zip'), env=virtual_env_list[0], tensorboard_log=LOG_DIR)
alg_1 = PPO.load(os.path.join(OPT_DIR, 'best_model_PPO_200000.zip'), env=virtual_env_list[1], tensorboard_log=LOG_DIR)

# Dictionary for DQN will specify DQN model itself, and obs_index 0 to pick first element from real environment return
alg_collection_0 = {}
alg_collection_0['alg'] = alg_0
alg_collection_0['env'] = virtual_env_list[0]
alg_collection_0['obs_index'] = 0
alg_collection_0['reward_index'] = 0
alg_collection_0['action_indexes'] = [0,1]
alg_collection_0['action_space'] = action_space_list[0]

# Dictionary for PPO will specify PPO model itself, and obs_index 0 to pick first element from real environment return
alg_collection_1 = {}
alg_collection_1['alg'] = alg_1
alg_collection_1['env'] = virtual_env_list[1]
alg_collection_1['obs_index'] = 0
alg_collection_1['reward_index'] = 1
alg_collection_1['action_indexes'] = [2,3]
alg_collection_1['action_space'] = action_space_list[1]

alg_collection = [alg_collection_0, alg_collection_1]



# Create MultiAlgorithm, this one is in contact with real environment and deffers actions and observation to virtual environments
model = MultiSB3(real_env, alg_collection, virtual_env_list)


count = 0


if(False):
    while(True):
        retVal, _, _, _, _ = real_env.step(None)
        real_env.render()
        count += 1
        if((count % 60) == 0 or True):
            pass

#observable_inst = pickle_skip.PickleSkipper(model)
#observable1 = observable_inst.Update()

#model.learn(total_timesteps=200000, callback_alg=callback_list)

model.evaluate_multipolicy(render=True, n_eval_episodes=15)

#env.close()