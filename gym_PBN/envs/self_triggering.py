import random
from typing import Tuple, Union

import numpy as np
from gymnasium import spaces

from gym_PBN.envs.pbcn_env import PBCNEnv
from gym_PBN.envs.dep_pbn_env import PBNEnv
from gym_PBN.types import GYM_STEP_RETURN
from gym_PBN.utils import booleanize

# Types
PBCN_MACRO_ACTION = Tuple[Tuple[Union[int, bool]], int]


class PBNSelfTriggeringEnv(PBNEnv):
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
        T: int = 5,
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
        self.T = T
        self.primitive_action_space = spaces.Discrete(self.PBN.N + 1)
        self.prob_space = spaces.Discrete(10, start=1)  # {0.1, 0.2, ..., 0.9}
        self.action_space = spaces.Tuple((self.primitive_action_space, self.prob_space))
        self.discrete_action_space = spaces.Discrete(
            self.primitive_action_space.n * self.prob_space.n
        )

        # Reward hardcode
        self.successful_reward = 1
        self.wrong_attractor_cost = 0
        self.action_cost = 1

    def step(self, action: Tuple[int, int]) -> GYM_STEP_RETURN:
        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        control_action, prob = action
        prob /= 10  # convert value in [1,10] to [0.1, 0.2, ..., 0.9, 1]

        total_reward, i, end = 0, 0, False
        while not end:
            # Take action
            if control_action != 0:  # Action 0 is taking no action.
                self.PBN.flip(control_action - 1)

            # Synchronous step / update in the network
            self.PBN.step()

            # Calculate reward
            observation = self.PBN.state
            reward, terminated, truncated = self._get_reward(
                observation, control_action
            )
            total_reward += (self.gamma**i) * reward
            i += 1
            end = random.uniform(0, 1) <= prob or i == self.T

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            {
                "control_action": control_action,
                "interval": i,
                "observation_idx": self._state_to_idx(observation),
                "T": self.T,
            },
        )


class PBCNSelfTriggeringEnv(PBCNEnv):
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

        self.T = T
        self.primitive_action_space = spaces.MultiBinary(self.PBN.M)
        self.primitive_action_space.dtype = bool
        self.prob_space = spaces.Discrete(10, start=1)  # {0.1, 0.2, ..., 0.9}
        self.action_space = spaces.Tuple((self.primitive_action_space, self.prob_space))
        self.discrete_action_space = spaces.Discrete(
            (2**self.primitive_action_space.n) * self.prob_space.n
        )

        # Reward hardcode
        self.successful_reward = 1
        self.wrong_attractor_cost = 1
        self.action_cost = 1

    def _idx_to_macro_action(self, i: int) -> PBCN_MACRO_ACTION:
        action = booleanize(
            i % (2**self.primitive_action_space.n), self.primitive_action_space.n
        ).tolist()
        prob_raw = i // (2**self.primitive_action_space.n) + 1
        return action, prob_raw

    def step(self, action: Union[PBCN_MACRO_ACTION, int]) -> GYM_STEP_RETURN:
        if action is None:
            raise Exception(
                "You need to provide a macro action with either `macro_action` or `macro_action_discrete`."
            )

        if np.isreal(action):
            if not self.discrete_action_space.contains(action):
                raise Exception(f"Invalid action {action}, not in action space.")

            action = self._idx_to_macro_action(action)

        if type(action[1]) is float:  # Adjust if float was passed in
            action = (action[0], int(action[1] * 10))

        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        control_action, prob = action
        prob /= 10  # convert value in [1,10] to [0.1, 0.2, ..., 0.9, 1]

        total_reward, i, end = 0, 0, False
        while not end:
            # Take action
            self.PBN.apply_control(control_action)

            # Synchronous step / update in the network
            self.PBN.step()

            # Calculate reward
            observation = self.PBN.state
            reward, terminated, truncated = self._get_reward(observation)
            reward -= 1  # Time step cost not in the original reward function
            total_reward += (self.gamma**i) * reward  # Internal reward discounting
            i += 1
            end = random.uniform(0, 1) <= prob or i == self.T

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            {
                "control_action": control_action,
                "interval": i,
                "observation_idx": self._state_to_idx(observation),
                "T": self.T,
            },
        )
