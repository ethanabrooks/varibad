import abc
import numpy as np
import re
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar
from enum import Enum, auto
from gym.envs.registration import load
from gym.wrappers import TimeLimit

import gym

from environments.icpi.wrapper import ArrayWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Data(Enum):
    code = auto()
    natural_language = auto()


@dataclass
class TimeStep(Generic[ObsType, ActType]):
    state: ObsType
    action: ActType
    reward: float
    done: bool
    next_state: ObsType


def create(entry_point, max_episode_steps, test_threshold=None, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs, hint=False)
    env._max_episode_steps = max_episode_steps
    return ArrayWrapper(TimeLimit(env, max_episode_steps=max_episode_steps))


@dataclass
class Env(gym.Env, Generic[ObsType, ActType], abc.ABC):
    hint: bool

    def action(self, action_str: Optional[str]) -> Optional[ActType]:
        action_space = self.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        try:
            actions = [
                a for a in range(action_space.n) if self.action_str(a) == action_str
            ]
            [action] = actions
            return action
        except ValueError:
            return None

    @staticmethod
    def action_stop() -> str:
        return "\n"

    @abc.abstractmethod
    def action_str(self, action: ActType) -> str:
        ...

    def done(self, done_str: str) -> bool:
        return "assert done" in done_str

    @staticmethod
    def done_stop() -> str:
        return "\n"

    def done_str(self, done: bool) -> str:
        return "assert done" if done else "assert not done"

    @abc.abstractmethod
    def failure_threshold(self) -> float:
        ...

    @staticmethod
    def gamma() -> float:
        return 0.8

    @staticmethod
    @abc.abstractmethod
    def initial_str() -> str:
        ...

    @classmethod
    def log_gamma(cls) -> float:
        return 1.0

    @abc.abstractmethod
    def max_q_steps(self) -> int:
        ...

    def render(self, mode="human"):
        pass

    @staticmethod
    def reward(reward_str: str) -> float:
        matches = re.findall(r"reward == (\d+)", reward_str)
        try:
            [reward] = matches
        except ValueError:
            return 0.0
        return float(reward)

    def reward_str(self, reward: float) -> str:
        return f"assert reward == {int(reward)}"

    def reward_stop(self) -> str:
        if self.data == Data.code:
            return "\n"
        elif self.data == Data.natural_language:
            return ". "
        raise RuntimeError("Invalid data")

    def state_stop(self) -> str:
        return "\n"

    @abc.abstractmethod
    def start_states(self) -> Optional[Iterable[ObsType]]:
        ...

    @abc.abstractmethod
    def state_str(self, state: ObsType) -> str:
        ...

    def termination_str(self, ts: TimeStep) -> str:
        return self.state_str(ts.next_state)

    def ts_to_string(self, ts: TimeStep) -> str:
        ...

    @abc.abstractmethod
    def valid_done(self, done_str: str) -> bool:
        ...

    @abc.abstractmethod
    def valid_reward(self, reward_str: str) -> bool:
        ...

    @abc.abstractmethod
    def valid_state(self, state_str: str) -> bool:
        ...

    @staticmethod
    def valid_transition(transition_str: str) -> bool:
        return (
            transition_str.startswith("assert")
            and "reward" in transition_str
            and "done" in transition_str
        )

    def transition_stop(self) -> str:
        return "done" + self.done_stop()

    def set_task(self, _):
        pass

    def get_task(self):
        return np.array([0])

    def reset_task(self, task):
        pass
