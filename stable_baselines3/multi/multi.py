import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Space
from gymnasium.spaces import Discrete, MultiBinary
from torch.nn import functional as F

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union, Callable
import time
from collections import deque
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
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
        
        self.next_step_obs = None
        self.next_reward = 0
        self.next_actions = None
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
        Obs, reward, done, trimmed, info which will be retrieved in designed algorithm policy rollout cycle.
        
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
        ) -> (List[int], bool, bool):
        """
        Gets last actions which step was called last time
        """
        next_reset = self.next_reset
        next_close = self.next_close
        self.next_reset = False
        self.next_close = False
        
        return self.next_actions, next_reset, next_close
    
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
    
    def clearReset(
        self: SelfMSB3_Venv,
        ) -> None:
        """
        Clears reset order given from algorithm

        """
        self.next_reset = False
        self.next_close = False
        
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
        return self.next_step_obs, self.next_reward, self.next_done, self.next_trimmed, self.next_info
    
    def close(
        self: SelfMSB3_Venv
        ) -> None:
        """
        Passes order to real Environment to close
        """
        self.next_close = True


class MultiSB3(BaseAlgorithm):
        """
        Multi Algorithm from SB3. This class wraps all inner classes contained in alg_collection.
        Distribution of Observation, Rewards, and Actions will be decided on their three associated keys.
        By the moment, MultiEnvironment are not supported as not all algorithms (like DQN) support this.
        Environment must be specially wrapped into additional array, as observation, actions, and rewards
        may be segmented for different uses, and actions are always expected as MultiBinary, therefore
        an algorithm could cope with one ore more actions.
        """
        @classmethod
        def createVirtualEnvironments(
                cls,
                numAlgorithms:int,
                observationSpaceList: List[Space],
                actionSpaceList: List[Space]
        ) -> List[SelfMSB3_Venv]:
            """
            Creates a list of virtual environments prepared for given observation and action spaces.
            Algorithms should be created/assigned with given Virtual environment, according to its order.
            :param numAlgorithms: Number of involved algorithms (does not matter type)
            :param observationSpaceList: List of Observation spaces ordered for algorithms
            :param actionSpaceList: List of Action spaces ordered for algorithms
            """
            retEnviornments = []
            
            for i in range(numAlgorithms):
                newVenv = MSB3_VirtualEnv(observationSpaceList[i], actionSpaceList[i])
                retEnviornments.append(newVenv)

            return retEnviornments
        
        def __init__(
            self,
            env: Union[GymEnv, str],
            alg_collection: List[Dict],
            virtual_env_list: List[SelfMSB3_Venv]
        ):
            """
            Initializes multi algorithm.
            
            :param env: The environment to learn from (if registered in Gym, can be str)
            :param alg_collection: List of involved parameters in training. It is expected a dictionary with 4 keys:
            'alg': Algorithm (PPO, SAC, DQN, ...) which must have been previously initialized with desired parameters
            'obs_index': Index from observation wrapping array to destinate for given algorithm. In case only
            one is available, choose always 0
            'reward_index': Index from rewards wrapping array to destinate for given algorithm. In case only
            one is available, choose always 0
            'action_indexes': List of indexes of actions which will be taken by algorithm from action_space
            'action_space': Action space associated to algorithm. Can be MultiBinary or Discrete
            according to action_space which MUST be MultiBinary.
            :param virtual_env_list: Previously created Virtual environment list with class method
            
            """
            
            super().__init__(
                policy=None,
                env=env,
                learning_rate=0.0
            )
            
            alg_collection_size = len(alg_collection)
            assert isinstance(env.action_space, spaces.MultiBinary), 'Action space must be MultiBinary'
            assert (len(virtual_env_list) == alg_collection_size), 'Virtual environment list size does not match algorithm list size'
            
            total_actions = env.action_space.shape[0]
            mask_total_actions = int((2**total_actions) - 1)
            observed_action_mask = int(0x0)
            
            alg_collection = alg_collection.copy()
            
            for alg_index in range(alg_collection_size):
                alg = alg_collection[alg_index]
                
                if(isinstance(alg['action_space'], Discrete)):
                    discrete = True
                elif(isinstance(alg['action_space'], MultiBinary)):
                    discrete = False
                else:
                    raise Exception('Action space should be Discrete or MultiBinary')
                    
                alg['discrete'] = discrete
                alg_action_list = alg['action_indexes']
                alg['action_indexes_range'] = range(len(alg_action_list))
                
                for i in alg_action_list:
                    prev_observed = observed_action_mask
                    observed_action_mask |= 1 << i
                    assert observed_action_mask != prev_observed, 'Some action is repeated or shared between algorithms'
                        
            assert observed_action_mask == mask_total_actions, 'Some action was not covered by algorithms'
            
            # Create a local copy of given input
            self.venv_list : List[MSB3_VirtualEnv] = virtual_env_list
            self.alg_collection = alg_collection
            self.total_actions : int = total_actions
            self.action_multibinary = [False] * total_actions
            self.no_reward = [0] * alg_collection_size
            self.pure_env = env
            
        def learn(
            self: SelfMSB3,
            total_timesteps: int,
            callback: MaybeCallback = None,
            callback_alg: List[MaybeCallback] = None,
            log_interval: int = 4,
            tb_log_name: str = "Multi",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
        )-> None:
            """
            Starts and completes whole learning process as if it was an individual algorithm.
            It iterates every cycle from real environment and feeding virtual environments.
            Each algorithm has only its relevant information and action range through steps.

            :param total_timesteps: Total timesteps taken from real environment
            :callback: Global operation callback (Optional, otherwise None)
            :callback_alg: (Optional, otherwise None). List of callbacks linked at rigurous order
            at which algorithms were declared on __init__. Single not interesting callbacks
            can be set to [None]
            :log_interval: Same as single algorithms
            :tb_log_name: Name for multi algorithm log
            :reset_num_timesteps: Same as single algorithms
            :progress_bar: Same as signel algorithms

            """
            
            alg_collection_size = len(self.alg_collection)
            # Initial per-algorithm-custom-callback check
            if(callback_alg is None):
                callback_alg = [None]*alg_collection_size
            elif(isinstance(callback_alg, list)):
                assert len(callback_alg) == alg_collection_size, 'Callback list does not match algorithm list size'
            else:
                assert False, 'Given callback list is not a list'
                
                
            # Initial reset
            obs, info = self.pure_env.reset()
            
            # Feed with initial observation and info as a result of real environment reset
            for alg in self.alg_collection:                
                alg['env'].feedNextStep(obs[alg['obs_index']], 0, False, False, info)
                
            # Every algorithm _setup_learn initialization
            for i in range(alg_collection_size):
                _, callback_alg[i] = self.alg_collection[i]['alg'].stepped_learn_start(\
                    total_timesteps, callback_alg[i], log_interval, tb_log_name+str(i),\
                    reset_num_timesteps, progress_bar)
                    
                # Clear reset order given by algorithms when they do _setup_learn(...)
                self.alg_collection[i]['env'].clearReset()
                
                # Store callback generated by _setup_lean
                self.alg_collection[i]['callback'] = callback_alg[i]
            
            # Create global callback if needed
            _, callback = self._setupCallback(total_timesteps=total_timesteps,\
                    callback = callback, progress_bar=progress_bar)
            callback = self._init_callback(callback, progress_bar)
            
            # Start event for given main callback
            callback.on_training_start(locals(), globals())
            
            # Set target and step num to configured initialization
            self.learn_step_num = 0
            self.learn_step_total = total_timesteps
            
            
            
            # Main Loop
            while(self.learn_step_num < self.learn_step_total):
                
                some_reset = False
                some_close = False
                
                callback.update_locals(locals())
                
                if not callback.on_step():
                    break
        
                callback.on_rollout_start()
                
                # Execute learn step sending from all algorithms
                for alg in self.alg_collection:
                    # Execute one sending step
                    alg['alg'].stepped_learn_send(alg['callback'])
                    
                    # Retrieve from virtual environment which action was proposed by algorithm
                    actions, reset, close = alg['env'].getLastActions()
                    
                    # Store reset or close orders
                    some_reset |= reset
                    some_close |= close
                    
                    # Fill actions
                    if(alg['discrete']):
                        for action_i in alg['action_indexes_range']:
                            action_n = alg['action_indexes'][action_i]
                            self.action_multibinary[action_n] = (actions == (1 << action_i))
                    else:
                        for action_i in alg['action_indexes_range']:
                            action_n = alg['action_indexes'][action_i]
                            self.action_multibinary[action_n] = actions[action_i]
                
                # Commit reset in case one of all algorithms asked for it (normally when real env was done, trimmed)
                if(some_reset):
                    obs, info = self.env.reset(None)
                    reward, done, trimmed = [self.no_reward, False, False]
                    for alg in self.alg_collection:
                        alg['env'].feedNextStep(obs[alg['obs_index']], reward[alg['reward_index']],\
                            done, trimmed, info)
                else:
                    # Otherwise, do a normal step in real environment with action multibinary array filled actions
                    obs, reward, done, trimmed, info = self.env.step(self.action_multibinary)
                    for alg in self.alg_collection:
                        alg['env'].feedNextStep(obs[alg['obs_index']], reward[alg['reward_index']],\
                            done, trimmed, info)
                        
                # Now it is time for receive step phase
                for alg in self.alg_collection:
                    alg.alg.stepped_learn_receive(alg['callback'])
                    
                callback.update_locals(locals())
                callback.on_rollout_end()
                    
            # Stepped learn end
            for alg in self.alg_collection:
                alg['alg'].stepped_learn_end(alg['callback'])
                
            callback.on_training_end()
            
            
        def _setupCallback(
            self: SelfMSB3,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False
        ) -> Tuple[int, BaseCallback]:
            """
            Initialize different variables needed for training.

            :param total_timesteps: The total number of samples (env steps) to train on
            :param callback: Callback(s) called at every step with state of the algorithm.
            :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
            :param tb_log_name: the name of the run for tensorboard log
            :param progress_bar: Display a progress bar using tqdm and rich.
            :return: Total timesteps and callback(s)
            """
            self.start_time = time.time_ns()

            if self.ep_info_buffer is None or reset_num_timesteps:
                # Initialize buffers if they don't exist, or reinitialize if resetting counters
                self.ep_info_buffer = deque(maxlen=self._stats_window_size)
                self.ep_success_buffer = deque(maxlen=self._stats_window_size)

            if self.action_noise is not None:
                self.action_noise.reset()

            if reset_num_timesteps:
                self.num_timesteps = 0
                self._episode_num = 0
            else:
                # Make sure training timesteps are ahead of the internal counter
                total_timesteps += self.num_timesteps
            self._total_timesteps = total_timesteps
            self._num_timesteps_at_start = self.num_timesteps


            # Configure logger's outputs if no logger was passed
            if not self._custom_logger:
                self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

            # Create eval callback if needed
            callback = self._init_callback(callback, progress_bar)

            return total_timesteps, callback
        
        def saveModel(
            self: SelfMSB3,
            model_index: int,
            model_path: str
        )-> None:
            """
            Saves model/algorithm with given index according to given model_path

            :param model_index: Index according to given algorithm list given at initialization
            :param model_path: Full path to store model. Normally actual trained steps are specified in name


            """
            assert model_index < len(self.alg_collection), 'Index exceeds stored number of models/algorithms'
            self.alg_collection[model_index].alg.model.save(model_path)

            
        def _setup_model(self) -> None:
            pass
        