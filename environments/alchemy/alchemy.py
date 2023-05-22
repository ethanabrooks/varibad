from typing import Optional

import gym
import numpy as np
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import utils
from dm_env import TimeStep
from gym import spaces


class AlchemyEnv(gym.Env):
    def __init__(
        self,
        seed: int,
        num_stones_per_trial: int,
        num_potions_per_trial: int,
        end_trial_action: bool,
        max_steps_per_trial: int,
        level_name: str = "alchemy/",
        **_
    ):
        self.env = symbolic_alchemy.get_symbolic_alchemy_level(
            level_name,
            seed=seed,
            num_trials=1,
            num_stones_per_trial=num_stones_per_trial,
            num_potions_per_trial=num_potions_per_trial,
            end_trial_action=end_trial_action,
            max_steps_per_trial=max_steps_per_trial,
            see_chemistries={
                "input_chem": utils.ChemistrySeen(
                    content=utils.ElementContent.GROUND_TRUTH
                )
            },
        )
        obs_spec = self.env.observation_spec()
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=obs_spec["symbolic_obs"].shape
        )
        self.action_space = spaces.Discrete(self.env.action_spec().maximum)
        [self.task_dim] = obs_spec["input_chem"].shape
        self._max_episode_steps = self.env.max_steps_per_trial
        self.belief_dim = 0
        self.num_states = None

    def reset_mdp(self):
        self.env._new_trial()
        return self.env.observation()["symbolic_obs"]

    def reset(self):
        timestep = self.env.reset()
        return self.observation_from_timestep(timestep)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        timestep = self.env.step(action)
        state = self.observation_from_timestep(timestep)
        action = self.env._int_to_slot_based_action(action)
        done_mdp = (
            action.end_trial
            or self.env._steps_this_trial >= self.env.max_steps_per_trial
        )
        info = {}
        info["done_mdp"] = done_mdp
        if self.env.is_new_trial():
            info["start_state"] = state
        assert np.array_equal(timestep.observation["input_chem"], self.get_task())
        return state, timestep.reward, timestep.last(), info

    def observation_from_timestep(self, timestep: TimeStep) -> np.ndarray:
        return timestep.observation["symbolic_obs"]

    def get_task(self) -> np.ndarray:
        return self.env.chem_observation()["input_chem"]

    def reset_task(self, task: Optional[np.ndarray]):
        assert task is None
        self.env.reset()


if __name__ == "__main__":
    env = AlchemyEnv(0)
    env.reset()
    done = False
    t = 0
    while not done:
        action = env.action_space.sample()
        breakpoint()
        obs, rew, done, info = env.step(action)
        print(obs)
        print(rew)
        print(done)
        print(info)
        t += 1
    breakpoint()
