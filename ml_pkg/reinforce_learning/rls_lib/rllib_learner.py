import os
import shutil
import numpy as np
import torch
from functools import partial
from gymnasium.spaces import Discrete, Box
import gymnasium as gym

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule

from ray.rllib.algorithms.bc.torch.default_bc_torch_rl_module import DefaultBCTorchRLModule
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.sac.torch.default_sac_torch_rl_module import DefaultSACTorchRLModule
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule

from rls_lib.rllib_callback import TensorboardCallback
from rls_lib.rllib_utils import RlPolicySettings, PPOPolicySettings, DQNPolicySettings, SacPolicySettings
from rls_lib.rllib_utils import NetworkSettings

"""
AlgorithmConfig:
    |- Algorithm
        |- EnvRunnerGroup
        |    |- EnvRunner (SingleAgentEnvRunner | MultiAgentEnvRunner)
        |        |- Env
        |        |- RlModule
        |- LearnerGroup
        |    |- Learner
        |       |- RlModule
        |       |- loss
        |       |- optimizer

Episodes: Store the trajectory of env and rl_module
"""

class PpoLearner(object):
    def __init__(self, learner_config: dict):
        self.learner_config = learner_config
        self.model_config: AlgorithmConfig = PPOConfig()
        self.model_algo: Algorithm = None
        self.rl_module: RLModule = None
        self.is_action_discrete: bool = None
        # self.default_TorchRLModule = DefaultBCTorchRLModule
        self.default_TorchRLModule: TorchRLModule = DefaultPPOTorchRLModule
    
    def generate_env(self, env_class: callable, env_config: dict) -> gym.Env:
        return env_class(**env_config)
    
    def setup(
        self, 
        env_dict: dict[str, callable | dict], 
        policy_settings: RlPolicySettings | PPOPolicySettings, 
        network_settings: NetworkSettings | None = None
    ):
        """
            env_dict: dict[str, callable | dict]
                env_class: callable
                env_config: dict
            custom_model_dict: dict[str, callable | dict]
                model_class: callable
                model_config: dict
        """
        _env = self.generate_env(env_dict["env_class"], env_dict["env_config"])
        if isinstance(_env.action_space, Discrete):
            self.is_action_discrete = True
        elif isinstance(_env.action_space, Box):
            self.is_action_discrete = False
        else:
            raise ValueError(f"Action space {_env.action_space} is not supported")
        
        self.model_config.framework(framework="torch")
        self.model_config.resources(
            num_gpus=self.learner_config.get("num_gpus", 0)
        )
        self.model_config.environment(
            env=env_dict["env_class"],
            env_config=env_dict["env_config"],
        )
        self.model_config.env_runners(
            num_env_runners=self.learner_config.get("num_env_runners", 1),
            num_envs_per_env_runner = self.learner_config.get("num_envs_per_env_runner", 1),
            num_cpus_per_env_runner = self.learner_config.get("num_cpus_per_env_runner", 1),
        )
        
        self.model_config.learners(
            num_learners=self.learner_config.get("num_learners", 1),  # Set this to greater than 1 to allow for DDP style updates.
            num_gpus_per_learner=self.learner_config.get("num_gpus_per_learner", 0),  # Set this to 1 to enable GPU training.
            num_cpus_per_learner=self.learner_config.get("num_cpus_per_learner", 1),
        )
        self.model_config.training(**policy_settings.to_dict())
        
        if network_settings is not None:
            if network_settings.model_class is not None:
                self.model_config.rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=network_settings.model_class,
                        observation_space=_env.observation_space,
                        action_space=_env.action_space,
                        # inference_only=False,
                        # learner_only=False,
                        model_config=network_settings.model_config,
                    )
                )
            else:
                """
                custom_model_dict["model_config"] is like: {"fcnet_hiddens": [128, 128], "fcnet_activation": "tanh"}
                """
                self.model_config.rl_module(
                    rl_module_spec=RLModuleSpec(
                        module_class=self.default_TorchRLModule,
                        observation_space=_env.observation_space,
                        action_space=_env.action_space,
                        model_config=network_settings.model_config,
                    )
                )
        
        self.model_config.callbacks(
            partial(TensorboardCallback, log_dir=self.learner_config['log_dir'])
        )
        
        self.model_algo = self.model_config.build()
        if self.learner_config.get("init_policy_dir", False):
            self.model_algo.from_checkpoint(self.learner_config["init_policy_dir"])        
        self.rl_module = self.model_algo.get_module()
    
    def train(self, num_epochs: int):
        for _ in range(num_epochs):
            self.model_algo.train()
    
    def compute_action(self, batch_obs: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(batch_obs, np.ndarray):
            batch_obs = torch.from_numpy(batch_obs).float()
        actions = self.rl_module.forward_inference({Columns.OBS: batch_obs})
        actions = actions.cpu().numpy()
        return actions
    
    def choose_action(self, batch_obs: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        action_logits = self.compute_action(batch_obs)
        if self.is_action_discrete:
            actions = np.argmax(action_logits, axis=1)
        else:
            actions = action_logits
        return actions, action_logits
    
    def save_state(self, checkpoint_dir: str):
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
        self.model_algo.save_to_path(checkpoint_dir)

    def restore_state(self, checkpoint_dir: str):
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist")
        self.model_algo.restore_from_path(checkpoint_dir)

    def save_rl_module(self, rl_module_dir: str):
        if os.path.exists(rl_module_dir):
            shutil.rmtree(rl_module_dir)
            os.makedirs(rl_module_dir)
        self.rl_module.save_to_path(rl_module_dir)

    def restore_rl_module(self, rl_module_dir: str):
        if not os.path.exists(rl_module_dir):
            raise FileNotFoundError(f"RL module directory {rl_module_dir} does not exist")
        self.rl_module = RLModule.from_checkpoint(rl_module_dir)
    
    def restore_algorithm(self, algorithm_dir: str):
        if not os.path.exists(algorithm_dir):
            raise FileNotFoundError(f"Algorithm directory {algorithm_dir} does not exist")
        self.model_algo = Algorithm.from_checkpoint(algorithm_dir)
    
    @staticmethod
    def get_config_from_rlmodule(rl_module: RLModule) -> dict:
        return rl_module.config


class DqnLearner(PpoLearner):
    def __init__(self, learner_config: dict):
        self.learner_config = learner_config
        self.model_config: AlgorithmConfig = DQNConfig()
        self.model_algo: Algorithm = None
        self.rl_module: RLModule = None
        self.is_action_discrete: bool = True
        self.default_TorchRLModule = DefaultDQNTorchRLModule
    
    def setup(
        self, 
        env_dict: dict[str, callable | dict], 
        policy_settings: RlPolicySettings | DQNPolicySettings, 
        network_settings: NetworkSettings | None = None
    ):
        _env = self.generate_env(env_dict["env_class"], env_dict["env_config"])
        assert isinstance(_env.action_space, Discrete), "DQN only supports discrete action spaces"
        return super().setup(env_dict, policy_settings, network_settings)


class SacLearner(PpoLearner):
    def __init__(self, learner_config: dict):
        self.learner_config = learner_config
        self.model_config: AlgorithmConfig = SACConfig()
        self.model_algo: Algorithm = None
        self.rl_module: RLModule = None
        self.is_action_discrete: bool = False
        self.default_TorchRLModule = DefaultSACTorchRLModule
        
    def setup(
        self, 
        env_dict: dict[str, callable | dict], 
        policy_settings: RlPolicySettings | SacPolicySettings, 
        network_settings: NetworkSettings | None = None
    ):
        _env = self.generate_env(env_dict["env_class"], env_dict["env_config"])
        assert isinstance(_env.action_space, Box), "SAC only supports continuous action spaces"

        self.model_config.framework(framework="torch")
        self.model_config.resources(
            num_gpus=self.learner_config.get("num_gpus", 0)
        )
        self.model_config.environment(
            env=env_dict["env_class"],
            env_config=env_dict["env_config"],
        )
        self.model_config.env_runners(
            num_env_runners=self.learner_config.get("num_env_runners", 1),
            num_envs_per_env_runner = self.learner_config.get("num_envs_per_env_runner", 1),
            num_cpus_per_env_runner = self.learner_config.get("num_cpus_per_env_runner", 1),
        )
        
        self.model_config.learners(
            num_learners=self.learner_config.get("num_learners", 1),  # Set this to greater than 1 to allow for DDP style updates.
            num_gpus_per_learner=self.learner_config.get("num_gpus_per_learner", 0),  # Set this to 1 to enable GPU training.
            num_cpus_per_learner=self.learner_config.get("num_cpus_per_learner", 1),
        )
        self.model_config.training(**policy_settings.to_dict())
        
        if network_settings is not None:
            assert network_settings.model_class is None, "Custom model class has not been implemented for SAC"
            ac_model_config = network_settings.model_config
            ac_model_config.update({
                "twin_q": True,  # must set to True for SAC
                # "fcnet_hiddens": [32, 16],
                # "fcnet_activation": "tanh",
            })
            self.model_config.rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=self.default_TorchRLModule,
                    observation_space=_env.observation_space,
                    action_space=_env.action_space,
                    model_config=ac_model_config
                )
            )
                
        self.model_config.callbacks(
            partial(TensorboardCallback, log_dir=self.learner_config['log_dir'])
        )
        
        self.model_algo = self.model_config.build()
        if self.learner_config.get("init_policy_dir", False):
            self.model_algo.from_checkpoint(self.learner_config["init_policy_dir"])        
        self.rl_module = self.model_algo.get_module()
