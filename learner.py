"""
Base Learner, without Meta-Learning.
Can be used to train for good average performance, or for the oracle environment.
"""

import time
from typing import Optional

import gym
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from wandb.sdk.wandb_run import Run

from algorithms.a2c import A2C
from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_num_updates(num_frames, policy_num_steps, num_processes):
    return int(num_frames) // policy_num_steps // num_processes


class Learner:
    """
    Learner (no meta-learning), can be used to train avg/oracle/belief-oracle policies.
    """

    def __init__(
        self,
        args,
        replay_buffers: Optional[list[ReplayBuffer]],
        logger: TBLogger,
        run: Optional[Run],
    ):
        print("Seed:", args.seed)
        self.args = args
        self.run = run

        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = get_num_updates(
            args.num_frames, args.policy_num_steps, args.num_processes
        )
        self.frames = 0
        self.iter_idx = -1

        self.logger = logger
        self.replay_buffers = replay_buffers

        # initialise environments
        print("Making environments...", end=" ")
        self.envs = make_vec_envs(
            env_name=args.env_name,
            seed=args.seed,
            num_processes=args.num_processes,
            gamma=args.policy_gamma,
            device=device,
            episodes_per_task=self.args.max_rollouts_per_task,
            normalise_rew=args.norm_rew_for_policy,
            ret_rms=None,
            tasks=None,
        )
        print("✓")

        if self.args.single_task_mode:
            # get the current tasks (which will be num_process many different tasks)
            self.train_tasks = self.envs.get_task()
            # set the tasks to the first task (i.e. just a random task)
            self.train_tasks[1:] = self.train_tasks[0]
            # make it a list
            self.train_tasks = [t for t in self.train_tasks]
            # re-initialise environments with those tasks
            self.envs = make_vec_envs(
                env_name=args.env_name,
                seed=args.seed,
                num_processes=args.num_processes,
                gamma=args.policy_gamma,
                device=device,
                episodes_per_task=self.args.max_rollouts_per_task,
                normalise_rew=args.norm_rew_for_policy,
                ret_rms=None,
                tasks=self.train_tasks,
            )
            # save the training tasks so we can evaluate on the same envs later
            utl.save_obj(
                self.train_tasks, self.logger.full_output_folder, "train_tasks"
            )
        else:
            self.train_tasks = None

        # calculate what the maximum length of the trajectories is
        args.max_trajectory_len = self.envs._max_episode_steps
        if self.args.max_rollouts_per_task is None:
            args.max_trajectory_len = None
        else:
            args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise policy
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        return OnlineStorage(
            args=self.args,
            num_steps=self.args.policy_num_steps,
            num_processes=self.args.num_processes,
            state_dim=self.args.state_dim,
            latent_dim=0,  # use metalearner.py if you want to use the VAE
            belief_dim=self.args.belief_dim,
            task_dim=self.args.task_dim,
            action_space=self.args.action_space,
            hidden_size=0,
            normalise_rewards=self.args.norm_rew_for_policy,
        )

    def initialise_policy(self):
        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=False,  # use metalearner.py if you want to use the VAE
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=0,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
        ).to(device)

        # initialise policy trainer
        if self.args.policy == "a2c":
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
            )
        elif self.args.policy == "ppo":
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
            )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """Main training loop"""
        start_time = time.time()

        # reset environments
        state, belief, task = utl.reset_env(self.envs, self.args)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)

        for self.iter_idx in range(self.num_updates):
            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):
                current_state = state

                # sample actions from policy
                with torch.no_grad():
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                    )

                # observe reward and next obs
                (
                    [state, belief, task],
                    (rew_raw, rew_normalised),
                    done,
                    infos,
                ) = utl.env_step(self.envs, action, self.args)
                next_state = state

                # create mask for episode ends
                masks_done = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                ).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor(
                    [
                        [0.0] if "bad_transition" in info.keys() else [1.0]
                        for info in infos
                    ]
                ).to(device)

                # reset environments that are done
                done_indices = np.argwhere(done.flatten()).flatten()
                if len(done_indices) > 0:
                    state, belief, task = utl.reset_env(
                        self.envs, self.args, indices=done_indices, state=state
                    )

                done_tensor = torch.from_numpy(np.array(done, dtype=float)).unsqueeze(1)

                # add experience to replay buffer
                if self.replay_buffers is not None:
                    done_mdp = torch.tensor([i["done_mdp"] for i in infos])
                    done_mdp = done_mdp[..., None]
                    source = dict(
                        state=current_state,
                        actions=action,
                        rewards=rew_raw,
                        done=done_tensor,
                        done_mdp=done_mdp,
                        next_state=next_state,
                    )
                    if task is not None:
                        source.update(task=task)
                    batch = TensorDict(source, batch_size=[self.args.num_processes])
                    for buffer, transition in zip(self.replay_buffers, batch):
                        buffer.add(transition)

                # add experience to policy buffer
                self.policy_storage.insert(
                    state=state,
                    belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=rew_raw,
                    rewards_normalised=rew_normalised,
                    value_preds=value,
                    masks=masks_done,
                    bad_masks=bad_masks,
                    done=done_tensor,
                )

                self.frames += self.args.num_processes

            # --- UPDATE ---

            train_stats = self.update(state=state, belief=belief, task=task)

            # log
            run_stats = [action, self.policy_storage.action_log_probs, value]
            if train_stats is not None:
                with torch.no_grad():
                    try:
                        self.log(run_stats, train_stats, start_time)
                    except (ConnectionResetError, BrokenPipeError) as e:
                        print(e)

            # clean up after update
            self.policy_storage.after_update()
        self.envs.close()

    def get_value(self, state, belief, task):
        return self.policy.actor_critic.get_value(
            state=state, belief=belief, task=task, latent=None
        ).detach()

    def update(self, state, belief, task):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:    policy_train_stats which are: value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch
        """
        # bootstrap next value prediction
        with torch.no_grad():
            next_value = self.get_value(state=state, belief=belief, task=task)

        # compute returns for current rollouts
        self.policy_storage.compute_returns(
            next_value,
            self.args.policy_use_gae,
            self.args.policy_gamma,
            self.args.policy_tau,
            use_proper_time_limits=self.args.use_proper_time_limits,
        )

        policy_train_stats = self.policy.update(policy_storage=self.policy_storage)

        return policy_train_stats, None

    def log(self, run_stats, train_stats, start):
        """
        Evaluate policy, save model, write to tensorboard logger.
        """

        # --- visualise behaviour of policy ---

        if (self.iter_idx + 1) % self.args.vis_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualise_behaviour(
                args=self.args,
                policy=self.policy,
                image_folder=self.logger.full_output_folder,
                iter_idx=self.iter_idx,
                ret_rms=ret_rms,
                tasks=self.train_tasks,
            )
            self.logger.save_pngs(self.iter_idx)

        # --- evaluate policy ----

        if (self.iter_idx + 1) % self.args.eval_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            returns_per_episode = utl_eval.evaluate(
                args=self.args,
                policy=self.policy,
                ret_rms=ret_rms,
                seed=self.iter_idx,
                tasks=self.train_tasks,
            )

            # log the average return across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add(
                    "return_avg_per_iter/episode_{}".format(k + 1),
                    returns_avg[k],
                    self.iter_idx,
                )
                self.logger.add(
                    "return_avg_per_frame/episode_{}".format(k + 1),
                    returns_avg[k],
                    self.frames,
                )
                self.logger.add(
                    "return_std_per_iter/episode_{}".format(k + 1),
                    returns_std[k],
                    self.iter_idx,
                )
                self.logger.add(
                    "return_std_per_frame/episode_{}".format(k + 1),
                    returns_std[k],
                    self.frames,
                )

            print(
                "Updates {}, num timesteps {}, FPS {} \n Mean return (train): {:.5f} \n".format(
                    self.iter_idx,
                    self.frames,
                    int(self.frames / (time.time() - start)),
                    returns_avg[-1].item(),
                )
            )

        # --- log some other things ---

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (
            train_stats is not None
        ):
            train_stats, _ = train_stats

            self.logger.add("policy_losses/value_loss", train_stats[0], self.iter_idx)
            self.logger.add("policy_losses/action_loss", train_stats[1], self.iter_idx)
            self.logger.add("policy_losses/dist_entropy", train_stats[2], self.iter_idx)
            self.logger.add("policy_losses/sum", train_stats[3], self.iter_idx)

            # writer.add_scalar('policy/action', action.mean(), j)
            self.logger.add(
                "policy/action", run_stats[0][0].float().mean(), self.iter_idx
            )
            if hasattr(self.policy.actor_critic, "logstd"):
                self.logger.add(
                    "policy/action_logstd",
                    self.policy.actor_critic.dist.logstd.mean(),
                    self.iter_idx,
                )
            self.logger.add("policy/action_logprob", run_stats[1].mean(), self.iter_idx)
            self.logger.add("policy/value", run_stats[2].mean(), self.iter_idx)

            param_list = list(self.policy.actor_critic.parameters())
            param_mean = np.mean(
                [
                    param_list[i].data.cpu().numpy().mean()
                    for i in range(len(param_list))
                ]
            )
            self.logger.add("weights/policy", param_mean, self.iter_idx)
            self.logger.add(
                "weights/policy_std", param_list[0].data.cpu().mean(), self.iter_idx
            )
            if param_list[0].grad is not None:
                param_grad_mean = np.mean(
                    [
                        param_list[i].grad.cpu().numpy().mean()
                        for i in range(len(param_list))
                    ]
                )
                self.logger.add("gradients/policy", param_grad_mean, self.iter_idx)
                self.logger.add(
                    "gradients/policy_std",
                    param_list[0].grad.cpu().numpy().mean(),
                    self.iter_idx,
                )
