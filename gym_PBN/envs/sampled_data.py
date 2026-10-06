from typing import Tuple, Union

import numpy as np
from gymnasium import spaces

from gym_PBN.envs.pbcn_env import PBCNEnv
from gym_PBN.envs.dep_pbn_env import PBNEnv
from gym_PBN.types import GYM_STEP_RETURN
from gym_PBN.utils import booleanize

# Types
PBCN_MACRO_ACTION = Tuple[Tuple[Union[int, bool]], int]


class PBNSampledDataEnv(PBNEnv):
    def __init__(
        self,
        render_mode: str = "human",
        render_no_cache: bool = False,
        PBN_data=[],
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
        gamma: float = 0.99,
        T: int = None,
    ):
        super().__init__(
            render_mode=render_mode,
            render_no_cache=render_no_cache,
            PBN_data=PBN_data,
            logic_func_data=logic_func_data,
            name=name,
            goal_config=goal_config,
            reward_config=reward_config,
        )

        # Params
        self.gamma = gamma

        # Gym
        self.T = T if T is not None else 2**self.PBN.N
        self.primitive_action_space = spaces.Discrete(self.PBN.N + 1)
        self.interval_space = spaces.Discrete(self.T, start=1)
        self.action_space = spaces.Tuple(
            (self.primitive_action_space, self.interval_space)
        )
        self.discrete_action_space = spaces.Discrete(
            self.primitive_action_space.n * self.interval_space.n
        )

    def step(self, action: Tuple[int, int]) -> GYM_STEP_RETURN:
        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        control_action, interval = action

        total_reward = 0
        for i in range(interval):
            # Take action
            if control_action != 0:  # Action 0 is taking no action.
                self.PBN.flip(control_action - 1)

            # Synchronous step / update in the network
            self.PBN.step()

            # Calculate reward
            # HACK This does not have any of the more nuanced reward function stuff
            # that "Sampled Data" PBCNs (a few lines below) have. This is because
            # this class ended up not being used in actual experiments for the paper at all,
            # and I didn't want to spend time on it.
            observation = self.PBN.state
            reward, terminated, truncated = self._get_reward(
                observation, control_action
            )
            total_reward += reward

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            {
                "control_action": control_action,
                "interval": i,
                "observation_idx": self._state_to_idx(observation),
            },
        )


class PBCNSampledDataEnv(PBCNEnv):
    def __init__(
        self,
        render_mode: str = "human",
        render_no_cache: bool = False,
        PBN_data=[],
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
        gamma: float = 0.99,
        T: int = None,
    ):
        super().__init__(
            render_mode,
            render_no_cache,
            PBN_data,
            logic_func_data,
            name,
            goal_config,
            reward_config,
        )

        # Params
        self.gamma = gamma

        # Update Gym
        self.observation_space = spaces.MultiBinary(self.PBN.N)
        self.observation_space.dtype = bool

        self.T = T if T is not None else 2**self.PBN.N
        self.primitive_action_space = spaces.MultiBinary(self.PBN.M)
        self.primitive_action_space.dtype = bool
        self.interval_space = spaces.Discrete(self.T, start=1)
        self.action_space = spaces.Tuple(
            (self.primitive_action_space, self.interval_space)
        )
        self.discrete_action_space = spaces.Discrete(
            (2**self.primitive_action_space.n) * self.interval_space.n
        )

    def _idx_to_macro_action(self, i: int) -> PBCN_MACRO_ACTION:
        action = booleanize(
            i % (2**self.primitive_action_space.n), self.primitive_action_space.n
        ).tolist()
        interval = i // (2**self.primitive_action_space.n) + 1
        return action, interval

    def step(self, action: Union[PBCN_MACRO_ACTION, int]) -> GYM_STEP_RETURN:
        if action is None:
            raise Exception(
                "You need to provide a macro action with either `macro_action` or `macro_action_discrete`."
            )

        if np.isreal(action):
            if not self.discrete_action_space.contains(action):
                raise Exception(f"Invalid action {action}, not in action space.")

            action = self._idx_to_macro_action(action)

        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        control_action, interval = action

        time_step_cost = 1

        total_reward, terminated_step = 0, None
        for i in range(interval):
            # Take action
            self.PBN.apply_control(control_action)

            # Synchronous step / update in the network
            self.PBN.step()

            # Calculate reward
            observation = self.PBN.state
            reward, terminated, truncated = self._get_reward(observation)
            reward -= time_step_cost

            # Penalize overshooting the attractor
            if terminated_step is not None:
                reward -= self.successful_reward
            elif terminated:
                terminated_step = i

            total_reward += reward

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            {
                "control_action": control_action,
                "interval": i + 1,
                "observation_idx": self._state_to_idx(observation),
            },
        )
