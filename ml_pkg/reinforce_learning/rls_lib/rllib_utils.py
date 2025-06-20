import ray
from ray.rllib.utils.typing import NotProvided
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


class NetworkSettings(object):
    def __init__(self, model_config: dict, model_class: callable | None = None):
        self.model_class = model_class
        self.model_config = model_config

    def to_dict(self) -> dict:
        return {
            "model_class": self.model_class,
            "model_config": self.model_config,
        }
    
    def check_model_config(self):
        if self.model_config is None:
            for key in self.model_config.keys():
                if key not in DefaultModelConfig.__dict__.keys():
                    raise ValueError(f"Invalid model config: {key}")


class RlPolicySettings(object):
    def __init__(
        self, 
        gamma: float = 0.99, 
        lr: float = 0.001, 
        train_batch_size_per_learner: int = 256, 
        minibatch_size: int = 64, 
        shuffle_batch_per_epoch: bool = False,
        grad_clip = NotProvided,
    ):
        self.gamma = gamma
        self.lr = lr
        self.train_batch_size_per_learner = train_batch_size_per_learner
        self.minibatch_size = minibatch_size
        self.shuffle_batch_per_epoch = shuffle_batch_per_epoch
        self.grad_clip = grad_clip

    def to_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "lr": self.lr,
            "train_batch_size_per_learner": self.train_batch_size_per_learner,
            "minibatch_size": self.minibatch_size,
            "shuffle_batch_per_epoch": self.shuffle_batch_per_epoch,
            "grad_clip": self.grad_clip,
        }


class PpoPolicySettings(RlPolicySettings):
    def __init__(
        self, 
        use_critic=False, 
        use_gae=True, 
        lambda_=1.0, 
        use_kl_loss=True, 
        kl_coeff=0.2, 
        kl_target=0.01, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_critic = use_critic
        self.use_gae = use_gae
        self.lambda_ = lambda_
        self.use_kl_loss = use_kl_loss
        self.kl_coeff = kl_coeff
        self.kl_target = kl_target

    def to_dict(self) -> dict:
        para_dict = super().to_dict
        para_dict.update({
            "use_critic": self.use_critic,
            "use_gae": self.use_gae,
            "lambda_": self.lambda_,
            "use_kl_loss": self.use_kl_loss,
            "kl_coeff": self.kl_coeff,
            "kl_target": self.kl_target,
        })
        return para_dict


class DqnPolicySettings(RlPolicySettings):
    def __init__(
        self, 
        target_network_update_freq: int = 500, 
        replay_buffer_type: str = "PrioritizedEpisodeReplayBuffer", 
        replay_buffer_capacity: int = 50000, 
        replay_buffer_alpha: float = 0.6, 
        replay_buffer_beta: float = 0.4, 
        noisy: bool = False, 
        dueling: bool = True, 
        double_q: bool = True, 
        hiddens: int = 128, 
        n_step: int = 1, 
        td_error_loss_fn: str = "mse", 
        categorical_distribution_temperature: float = 0.9, 
        tau: float = 0.05, 
        num_steps_sampled_before_learning_starts: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_network_update_freq = target_network_update_freq
        self.replay_buffer_config = {
            "type": replay_buffer_type,
            "capacity": replay_buffer_capacity,
            "alpha": replay_buffer_alpha,
            "beta": replay_buffer_beta,
        }
        self.noisy = noisy
        self.dueling = dueling
        self.double_q = double_q
        self.hiddens = hiddens
        self.n_step = n_step
        self.td_error_loss_fn = td_error_loss_fn
        self.categorical_distribution_temperature = categorical_distribution_temperature
        self.tau = tau
        self.num_steps_sampled_before_learning_starts = num_steps_sampled_before_learning_starts

    @property
    def to_dict(self) -> dict:
        para_dict = super().to_dict
        para_dict.update({
            "target_network_update_freq": self.target_network_update_freq,
            "replay_buffer_config": self.replay_buffer_config,
            "noisy": self.noisy,
            "dueling": self.dueling,
            "double_q": self.double_q,
            "hiddens": self.hiddens,
            "n_step": self.n_step,
            "td_error_loss_fn": self.td_error_loss_fn,
            "categorical_distribution_temperature": self.categorical_distribution_temperature,
            "tau": self.tau,
            "num_steps_sampled_before_learning_starts": self.num_steps_sampled_before_learning_starts,
        })
        return para_dict


class SacPolicySettings(RlPolicySettings):
    def __init__(
        self, 
        tau: float = 0.05, 
        n_step: int = 1, 
        replay_buffer_type: str = "PrioritizedEpisodeReplayBuffer", 
        replay_buffer_capacity: int = 50000, 
        replay_buffer_alpha: float = 0.6, 
        replay_buffer_beta: float = 0.4, 
        actor_lr: float = 0.001, 
        critic_lr: float = 0.001, 
        target_network_update_freq: int = 500, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.n_step = n_step
        self.replay_buffer_config = {
            "type": replay_buffer_type,
            "capacity": replay_buffer_capacity,
            "alpha": replay_buffer_alpha,
            "beta": replay_buffer_beta,
        }
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_network_update_freq = target_network_update_freq

    def to_dict(self) -> dict:
        para_dict = super().to_dict
        para_dict.update({
            "tau": self.tau,
            "n_step": self.n_step,
            "replay_buffer_config": self.replay_buffer_config,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "target_network_update_freq": self.target_network_update_freq,
        })
        return para_dict
