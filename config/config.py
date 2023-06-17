import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Args:
    # training parameters
    # --- GENERAL ---
    num_frames: int = field(init=False)  # number of frames to train
    max_rollouts_per_task: int = 1
    exp_label = None  # label for the experiment
    env_name: str = field(init=False)  # environment to train on
    disable_metalearner: bool = (
        True  # Train a normal policy without the variBAD architecture
    )
    single_task_mode: bool = (
        False  # train policy on one (randomly chosen environment only)
    )
    test_threshold: Optional[float] = None  # parameter for setting test env

    # --- POLICY ---
    # what to pass to the policy (note this is after the encoder
    pass_state_to_policy: bool = False  # condition policy on state
    pass_latent_to_policy: bool = False  # condition policy on VAE latent
    pass_belief_to_policy: bool = False  # condition policy on ground-truth belief
    pass_task_to_policy: bool = (
        False  # condition policy on ground-truth task description
    )

    # using separate encoders for the different inputs (None uses no encoder
    policy_state_embedding_dim: int = None
    policy_latent_embedding_dim: int = None
    policy_belief_embedding_dim: int = None
    policy_task_embedding_dim: int = None

    # normalising (inputs/rewards/outputs
    norm_state_for_policy: bool = False  # normalise state input
    norm_latent_for_policy: bool = False  # normalise latent input
    norm_belief_for_policy: bool = False  # normalise belief input
    norm_task_for_policy: bool = False  # normalise task input
    norm_rew_for_policy: bool = False  # normalise rew for RL train
    norm_actions_pre_sampling: bool = False  # normalise policy output
    norm_actions_post_sampling: bool = False  # normalise policy output

    # network
    policy_layers: list[int] = field(default_factory=lambda: [128, 128])
    policy_activation_function: str = "tanh"  # tanh/relu/leaky-relu
    policy_initialisation: str = "normc"  # normc/orthogonal
    policy_anneal_lr: bool = False

    # RL algorithm
    policy: str = "ppo"  # choose: a2c ppo
    policy_optimiser: str = "adam"  # choose: rmsprop adam

    # PPO specific
    ppo_num_epochs: int = 2  # number of epochs per PPO update
    ppo_num_minibatch: int = 2  # number of minibatches to split the data
    ppo_use_huberloss: bool = True  # use huberloss instead of MSE
    ppo_use_clipped_value_loss: bool = True  # clip value loss
    ppo_clip_param: float = 0.1  # clamp param

    # other hyperparameters
    lr_policy: float = 7e-4  # learning rate (: 7e-4)
    num_processes: int = (
        16  # how many training CPU processes / parallel environments to use (: 16)
    )
    policy_num_steps: int = (
        200  # number of env steps to do (per process before updating)
    )
    policy_eps: float = 1e-8  # optimizer epsilon (1e-8 for ppo 1e-5 for a2c)
    policy_init_std: float = 1.0  # only used for continuous actions
    policy_value_loss_coef: float = 0.5  # value loss coefficient
    policy_entropy_coef: float = 0.01  # entropy term coefficient
    policy_gamma: float = 0.97  # discount factor for rewards
    policy_use_gae: bool = True  # use generalized advantage estimation
    policy_tau: float = 0.9  # gae parameter
    use_proper_time_limits: bool = (
        True  # treat timeout and death differently (important in mujoco)
    )
    policy_max_grad_norm: float = 0.5  # max norm of gradients

    # --- OTHERS ---

    # logging saving evaluation
    log_interval: int = 25  # log interval one log per n updates
    save_interval: int = 500  # save interval one save per n updates
    save_intermediate_models: bool = False  # save all models
    eval_interval: int = 25  # eval interval one eval per n updates
    vis_interval: int = 500  # visualisation interval one eval per n updates
    results_log_dir = os.environ["RESULTS_LOG_DIR"]  # directory to save results
    seed: list[int] = field(default_factory=lambda: [73])
    # general settings
    deterministic_execution: bool = False  # Make code fully deterministic. Expects 1 process and uses deterministic CUDNN


@dataclass
class Expert(Args):
    exp_label: str = "expert"
    single_task_mode: bool = True
    pass_state_to_policy: bool = True
    norm_state_for_policy: bool = True
    norm_latent_for_policy: bool = True
    norm_belief_for_policy: bool = True
    norm_rew_for_policy: bool = True


@dataclass
class Multitask(Args):
    exp_label: str = "multitask"
    norm_state_for_policy: bool = True
    norm_latent_for_policy: bool = True
    norm_belief_for_policy: bool = True
    norm_rew_for_policy: bool = True
    pass_state_to_policy: bool = True
    pass_task_to_policy: bool = True
    policy_num_steps: int = 400
    policy_state_embedding_dim: int = 32
    policy_task_embedding_dim: int = 32
    ppo_num_minibatch: int = 4


@dataclass
class AD(Multitask):
    exp_label: str = "AD"
    max_rollouts_per_task: int = None


@dataclass
class RL2(Args):
    exp_label: str = "rl2"
    num_train_tasks: Optional[int] = None
    num_test_tasks: Optional[int] = None
    policy_layers: list[int] = field(default_factory=lambda: [128])
    disable_decoder: bool = True
    disable_kl_term: bool = True
    add_nonlinearity_to_latent: bool = True
    rlloss_through_encoder: bool = True
    latent_dim: int = 128
    pass_latent_to_policy: bool = True
    norm_rew_for_policy: bool = True
    ppo_clip_param: float = 0.05
    lr_vae: float = 7e-4
    encoder_max_grad_norm: float = 0.5
    decoder_max_grad_norm: float = None
    size_vae_buffer: int = 0
    precollect_len: int = 0
    vae_buffer_add_thresh: float = 1
    vae_batch_num_trajs: int = 10  # how many trajectories to use for VAE update
    tbptt_stepsize: int = None  # stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)
    vae_subsample_elbos: int = (
        50  # for how many timesteps to compute the ELBO; None uses all
    )
    vae_subsample_decodes: int = (
        None  # number of reconstruction terms to subsample; None uses all
    )
    num_vae_updates: int = 1  # how many VAE update steps to take per meta-iteration
    pretrain_len: int = 0  # for how many updates to pre-train the VAE
    kl_weight: float = 1.0  # weight for the KL term"
    split_batches_by_task: bool = False  # split batches up by task (to save memory or if tasks are of different length)
    split_batches_by_elbo: bool = False  # split batches up by elbo term (to save memory of if ELBOs are of different length)

    # - encoder
    action_embedding_size: int = 16
    state_embedding_size: int = 32
    reward_embedding_size: int = 16
    encoder_layers_before_gru: list[int] = field(default_factory=list)
    encoder_gru_hidden_size: int = 128  # dimensionality of RNN hidden state
    encoder_layers_after_gru: list[int] = field(default_factory=list)

    # - decoder: rewards
    decode_reward: bool = False  # use reward decoder
    rew_loss_coeff: float = 1.0  # weight for state loss (vs reward loss)
    input_prev_state: bool = True  # use prev state for rew pred
    input_action: bool = True  # use prev action for rew pred
    reward_decoder_layers: list[int] = field(default_factory=lambda: [64, 32])
    multihead_for_reward: bool = False  # one head per reward pred (i.e. per state)
    rew_pred_type: str = "deterministic"  # choose: " "bernoulli (predict p(r=1|s))" "categorical (predict p(r=1|s) but use softmax instead of sigmoid)" "deterministic (treat as regression problem)"

    # - decoder: state transitions
    decode_state: bool = False  # use state decoder"
    state_loss_coeff: float = 1.0  # weight for state loss"
    state_decoder_layers: list[int] = field(default_factory=lambda: [64, 32])
    state_pred_type: str = "deterministic"  # choose: deterministic, gaussian

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    decode_task: bool = False  # use task decoder"
    task_loss_coeff: float = 1.0  # weight for task loss"
    task_decoder_layers: int = field(default_factory=lambda: [64, 32])
    task_pred_type: str = "task_id"  # choose: task_id, task_description

    # --- ABLATIONS ---

    # for the policy training
    sample_embeddings: bool = (
        False  # sample embedding for policy, instead of full belief
    )

    # combining vae and RL loss
    vae_loss_coeff: float = 1.0  # weight for VAE loss (vs RL loss)

    # for other things
    disable_metalearner: bool = False  # Train feedforward policy
    single_task_mode: bool = (
        False  # train policy on one (randomly chosen) environment only
    )


@dataclass
class Alchemy(Args):
    env_name: str = "Alchemy-v0"
    num_frames: int = 5e7


@dataclass
class AlchemyExpert(Alchemy, Expert):
    pass


@dataclass
class AlchemyMultitask(Alchemy, Multitask):
    pass


@dataclass
class AlchemyAD(AD, AlchemyMultitask):
    pass


@dataclass
class AlchemyRL2(RL2, Alchemy):
    pass


@dataclass
class AntDir(Args):
    num_frames: int = 1e8
    env_name: str = "AntDir2D-v0"


@dataclass
class AntDirExpert(Expert, AntDir):
    pass


@dataclass
class AntDirMultitask(Multitask, AntDir):
    policy_num_steps: int = 200
    policy_state_embedding_dim: int = 64
    policy_task_embedding_dim: int = 64
    ppo_num_minibatch: int = 2


@dataclass
class AntDirAD(AD, AntDirMultitask):
    pass


@dataclass
class AntDirRL2(RL2, AntDir):
    max_rollouts_per_task: int = 2
    kl_weight: float = 0.1
    ppo_num_minibatch: int = 4


@dataclass
class AntGoal(Args):
    env_name: str = "AntGoal-v0"
    num_frames: int = 1e8


@dataclass
class AntGoalExpert(Expert, AntGoal):
    lr_policy: float = 0.001
    policy_anneal_lr: bool = True
    policy_initialisation: str = "orthogonal"
    vis_interval: int = 1


@dataclass
class AntGoalMultitask(Multitask, AntGoal):
    lr_policy: float = 0.001
    policy_anneal_lr: bool = True
    policy_initialisation: str = "orthogonal"
    policy_num_steps: int = 200
    policy_state_embedding_dim: int = 64
    policy_task_embedding_dim: int = 64
    ppo_num_minibatch: int = 2


@dataclass
class AntGoalAD(AD, AntGoalMultitask):
    num_frames: int = 30_000_000


@dataclass
class AntGoalRL2(RL2, AntGoal):
    kl_weight: float = 0.1
    lr_policy: float = 0.0003
    lr_vae: float = 0.0003
    max_rollouts_per_task: int = 2
    num_frames: int = 2e8
    policy_anneal_lr: bool = True
    policy_initialisation: str = "orthogonal"
    policy_latent_embedding_dim: int = 64
    policy_state_embedding_dim: int = 64
    ppo_num_minibatch: int = 1
    vae_subsample_elbos: int = None


@dataclass
class CheetahDir(Args):
    env_name: str = "HalfCheetahDir-v0"
    num_frames: int = 1e8


@dataclass
class CheetahDirExpert(Expert, CheetahDir):
    lr_policy: float = 0.001
    policy_anneal_lr: bool = True
    policy_num_steps: int = 800
    ppo_num_epochs: int = 16
    ppo_num_minibatch: int = 4


@dataclass
class CheetahDirMultitask(Multitask, CheetahDir):
    lr_policy: float = 0.001
    policy_anneal_lr: bool = True
    policy_num_steps: int = 800
    ppo_num_epochs: int = 16


@dataclass
class CheetahDirAD(AD, CheetahDirMultitask):
    pass


@dataclass
class CheetahDirRL2(RL2, CheetahDir):
    kl_weight: float = 0.1
    max_rollouts_per_task: int = 2
    ppo_clip_param: float = 0.1
    ppo_num_minibatch: int = 4


@dataclass
class CheetahVel(Args):
    env_name: str = "HalfCheetahVel-v0"
    num_frames: int = 1e8


@dataclass
class CheetahVelExpert(Expert, CheetahVel):
    policy_num_steps: int = 400
    ppo_num_epochs: int = 16
    ppo_num_minibatch: int = 4


@dataclass
class CheetahVelMultitask(Multitask, CheetahVel):
    policy_state_embedding_dim: int = 64
    policy_task_embedding_dim: int = 64
    ppo_num_epochs: int = 16


@dataclass
class CheetahVelAD(AD, CheetahVelMultitask):
    pass


@dataclass
class CheetahVelRL2(RL2, CheetahVel):
    kl_weight: float = 0.1
    max_rollouts_per_task: int = 2
    policy_num_steps: int = 800
    ppo_clip_param: float = 0.1
    ppo_num_minibatch: int = 4


@dataclass
class PointRobot(Args):
    env_name: str = "SparsePointEnv-v0"
    num_frames: int = 5e7


@dataclass
class PointRobotAD(AD, PointRobot):
    pass


@dataclass
class PointRobotExpert(Expert, PointRobot):
    pass


@dataclass
class PointRobotMultitask(Multitask, PointRobot):
    pass


@dataclass
class PointRobotRL2(RL2, PointRobot):
    max_rollouts_per_task: int = 3
    policy_entropy_coef: float = 0.001
    policy_gamma: float = 0.99
    policy_num_steps: int = 600
    ppo_clip_param: float = 0.1
    ppo_num_minibatch: int = 8
    vae_subsample_elbos: int = None
    vae_avg_elbo_terms: bool = False
    vae_avg_reconstruction_terms: bool = False


@dataclass
class Walker(Args):
    env_name: str = "Walker2DRandParams-v0"
    num_frames: int = 1e8


@dataclass
class WalkerExpert(Expert, Walker):
    norm_task_for_policy: bool = True
    ppo_clip_param: float = 0.05
    num_frames: int = 10_000_000


@dataclass
class WalkerMultitask(Multitask, Walker):
    results_log_dir: None = None
    policy_state_embedding_dim: int = 128
    policy_task_embedding_dim: int = 128
    norm_task_for_policy: bool = True
    policy_layers: list = field(default_factory=lambda: [256, 128])
    ppo_num_minibatch: int = 2
    ppo_clip_param: float = 0.05
    policy_num_steps: int = 200


@dataclass
class WalkerAD(AD, WalkerMultitask):
    pass


@dataclass
class WalkerRL2(RL2, Walker):
    max_rollouts_per_task: int = 2
    ppo_num_minibatch: int = 1
    kl_weight: float = 0.1


@dataclass
class Hopper(Args):
    env_name: str = "HopperRandParams-v0"
    num_frames: int = 1e8


@dataclass
class HopperExpert(Hopper, WalkerExpert):
    pass


@dataclass
class HopperMultitask(Hopper, WalkerMultitask):
    pass


@dataclass
class HopperRL2(Hopper, WalkerRL2):
    pass
