import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Space
from torch.nn import functional as F

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Callable

from stable_baselines3.common import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfMSB3_Venv = TypeVar("SelfMSB3_Venv", bound="MSB3_VirtualEnv")
SelfMSB3 = TypeVar("SelfMSB3", bound="MultiSB3")

class MSB3_VirtualEnv(Env):
    """
    Virtual environment for every involved algorithm inside MultiSB3. This would be automatically created
    when MultiSB3 instance is initialized. It will emulate original environment AS if it was only yielding
    observation and rewards designed for a given algorithm, and will accept only actions designed for algorithm.
    Furthermore, it will launch reset and designed actions to main environment.
    """
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        render_mode:str = 'human',
    ):
        super().__init__()
        
        #Parent class needed
        self.render_mode = render_mode
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.next_step_obs = np.zeros(1)
        self.next_reward = 0
        self.next_actions = np.zeros(1)
        self.next_reset = False
        self.next_close = False
        self.next_done = False
        self.next_trimmed = False
        self.next_info = {}
        
        self.first_feed_done = False
        self.first_obs = np.zeros(1)
        self.first_info = {}
        
    def feedNextStep(
        self: SelfMSB3_Venv,
        next_obs: np.ndarray,
        next_reward: int,
        next_done: bool,
        next_trimmed: bool,
        next_info: Dict
        )->None:
        """
        Prepares next step with selected params for associated algorithm.
        Obs, reward, done, trimmed, info which will be retrieved in designed algorithm policy rollout cycle
        :param next_obs: Next obsevation matrix, box, array or space which was defined
        :param next_reward: Next reward specially designed for designed algorithm learning
        :param next_done: Next done boolean telling chapter was ended in real environment
        :param next_trimmed: Next trimmed boolean telling chapter was ended abruptly in real environment
        :param next_info: Next info
        """
        self.next_step_obs = next_obs
        self.next_reward = next_reward
        self.next_done = next_done
        self.next_trimmed = next_trimmed
        self.next_info = next_info
        
        if(not self.first_feed_done):
            self.first_feed_done = True
            self.first_obs = next_obs.copy()
            self.first_info = next_info.copy()
        
    def getLastActions(
        self: SelfMSB3_Venv
        ) -> (List[int], bool):
        """
        Gets last actions which step was called last time
        """
        next_reset = self.next_reset
        self.next_reset = False
        
        return self.next_actions, next_reset
    
    def reset(
        self: SelfMSB3_Venv,
        seed:int|None = None 
        ) -> None:
        """
        Emulates reset order, this can be retrieved inside external Env and process it with getLastActions
        :param seed: Random Seed [Optional]
        """
        super().reset(seed=seed)
        
        self.next_reset = True
        
        return self.next_step_obs, self.next_info
        
    def render(
        self: SelfMSB3_Venv,
        *args,
        **kwargs
        ) -> None:
        """
        Render has no special effect for this kind of Virtual environments. But its presence is required
        """
        pass
    
    def step(
        self: SelfMSB3_Venv,
        actions: List[int]
        ) -> (np.ndarray, float, bool, bool, Dict):
        """
        Angular stone of Virtual Environment. This stores proposed actions from its associated algorithm.
        And this actions can be retrieved externally to be carried to the real Environment (in the correct place).
        It also returns observer which was feeded from external Environment previously
        :param actions: Actions proposed from Policy from associated algorithm
        """
        self.next_actions = actions
        
        #Return observation, reward, done, trimmed, info
        return self.next_step_obs, self.next_done, self.next_trimmed, self.next_info
    
    def close(
        self: SelfMSB3_Venv
        ) -> None:
        """
        Passes order to real Environment to close
        """
        self.next_close = True


class MultiSB3:
        """
        Multi Algorithm from SB3. This class wraps all inner classes contained in alg_collection.
        Distribution of Observation, Rewards, and Actions will be decided on their three associated keys.
        By the moment, MultiEnvironment are not supported as not all algorithms (like DQN) support this.
        Environment must be specially wrapped into additional array, as observation, actions, and rewards
        may be segmented for different uses, and actions are always expected as MultiBinary, therefore
        an algorithm could cope with one ore more actions.
        """
        def __init__(
            self,
            env: Union[GymEnv, str],
            alg_collection: List[Dict[BaseAlgorithm,int,int,List[bool]]]
        ):
            """
            :param env: The environment to learn from (if registered in Gym, can be str)
            :param alg_collection: List of involved parameters in training. It is expected a dictionary with 4 keys:
            'alg': Algorithm (PPO, SAC, DQN, ...) which must have been previously initialized with desired parameters
            'obs_index': Index from observation wrapping array to destinate for given algorithm. In case only
            one is available, choose always 0
            'reward_index': Index from rewards wrapping array to destinate for given algorithm. In case only
            one is available, choose always 0
            'action_indexes': Boolean list [size of action_space] to indicate algorithm controlls given action/actions
            according to action_space which MUST be MultiBinary.
            """
            pass
        