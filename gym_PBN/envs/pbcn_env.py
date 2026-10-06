from typing import Tuple, Union

import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary

from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED
from gym_PBN.utils import booleanize

from .common.pbcn import PBCN
from .dep_pbn_env import PBNEnv


class PBCNEnv(PBNEnv):
    metadata = {"render_modes": ["human", "PBN", "STG", "funcs", "idx", "float"]}

    def __init__(
        self,
        render_mode: str = "human",
        render_no_cache: bool = False,
        PBN_data=[],
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
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

        # Switch to PBCN
        self.PBN = PBCN(PBN_data, logic_func_data)

        # Update Gym
        self.observation_space = MultiBinary(self.PBN.N)
        self.observation_space.dtype = bool
        self.action_space = MultiBinary(self.PBN.M)
        self.action_space.dtype = bool
        self.discrete_action_space = Discrete(2**self.action_space.n)
        self.target_nodes = goal_config["target_nodes"]

    def getTargetIdx(self):
        state = self.PBN.state
        return int(tuple(state) in self.target_nodes)

    def _get_reward(self, observation: STATE) -> Tuple[REWARD, TERMINATED, TRUNCATED]:
        reward, terminated, truncated = 0, False, False
        observation_tuple = tuple(observation)

        if observation_tuple in self.target_nodes:
            reward += self.successful_reward
            terminated = True
        else:
            attractors_matched = sum(
                observation_tuple in attractor for attractor in self.all_attractors
            )
            reward -= self.wrong_attractor_cost * attractors_matched

        return reward, terminated, truncated

    def step(self, action: int = 0) -> GYM_STEP_RETURN:
        # if not self.action_space.contains(action):
        #     raise Exception(f"Invalid action {action}, not in action space.")

        if action != 0:
            self.PBN.flip(action)

        self.PBN.step()

        observation = self.PBN.state
        reward, terminated, truncated = self._get_reward(observation)
        info = {"observation_idx": self._state_to_idx(observation)}

        return observation, reward, terminated, truncated, info

    # def compute_attractors(self):
    #     attractor_sets = []
    #     for action in self.PBN.control_actions:
    #         print(f"Computing attractors for action {action}...")
    #         self.PBN.apply_control(action)
    #         STG = self.render(mode="STG", no_cache=True)
    #         generator = nx.algorithms.components.attracting_components(STG)
    #         attractors = self._nx_attractors_to_tuples(list(generator))
    #         attractor_sets.append((action, attractors))
    #     return attractor_sets
