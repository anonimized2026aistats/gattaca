from typing import List, Set, Tuple, Union
import random

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, MultiBinary
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED

from .common.pbn import PBN


class PBNEnv(gym.Env):
    metadata = {"render_modes": ["human", "PBN", "STG", "funcs", "idx", "float"]}

    def set(self, new_state):
        self.PBN.state = np.array(new_state)

    def __init__(
        self,
        render_mode: str = "human",
        render_no_cache: bool = False,
        PBN_data=None,
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
    ):
        if PBN_data is None:
            PBN_data = []

        print("normal pbn")
        self.PBN = PBN(PBN_data, logic_func_data)

        # Goal configuration
        goal_config = self._check_config(
            goal_config, "goal", {"target", "all_attractors"}
        )
        if (
            goal_config is None
        ):  # If no goal config is provided, then compute attractors and set the target as the last attractor.
            goal_config = {}
            goal_config["all_attractors"] = self.compute_attractors()
            goal_config["target"] = goal_config["all_attractors"][-1]
        else:
            assert (
                type(goal_config["target_nodes"]) is set
            ), "Did you put multiple attractors as the target by mistake?"
        # self.all_attractors = goal_config["all_attractors"]
        self.all_attractors = self.compute_attractors()
        self.target_nodes = goal_config["target_nodes"]

        for attractor in self.all_attractors:
            if self.target_nodes & attractor:
                self.target_nodes = self.target_nodes.union(attractor)

        print(f"target nodes are {self.target_nodes}")

        self.attracting_states = set.union(*self.all_attractors)

        # Reward configuration
        reward_config = self._check_config(
            reward_config,
            "reward",
            {"successful_reward", "wrong_attractor_cost", "action_cost"},
            default_values={
                "successful_reward": 10,
                "wrong_attractor_cost": 2,
                "action_cost": 1,
            },
        )
        self.successful_reward = reward_config["successful_reward"]
        self.wrong_attractor_cost = reward_config["wrong_attractor_cost"]
        self.action_cost = reward_config["action_cost"]

        # Gym
        self.observation_space = MultiBinary(self.PBN.N)
        self.observation_space.dtype = bool
        self.action_space = Discrete(self.PBN.N)
        self.name = name
        self.render_mode = render_mode
        self.render_no_cache = render_no_cache
        self.step_no = 0

    def _seed(self, seed: int = None):
        np.random.seed(seed)
        random.seed(seed)

    def _check_config(
        self,
        config: dict,
        _type: str,
        required_keys: Set[str],
        default_values: dict = None,
    ) -> dict:
        """Small utility function to validate an environment config.

        Args:
            config (dict): The config to validate.
            _type (str): The type of config this is about. Just needs to be a semantically rich string for the exception output.
            required_keys (Set[str]): The mandatory keys that need to be in the config.
            default_values (dict, optional): The default values for the config should it be empty. Defaults to None.

        Raises:
            ValueError: Thrown when some or all of the required keys are missing in the given config values.

        Returns:
            dict: The config after it has been checked and initialized to default values if it was empty to begin with.
        """
        if config:
            missing_keys = required_keys - set(config.keys())
            if len(missing_keys) > 1:  # If any of the required keys are missing
                raise ValueError(
                    f"Invalid {_type} config provided. The following required values are missing: {', '.join(missing_keys)}."
                )
        else:
            config = default_values

        return config

    def step(self, action: int) -> GYM_STEP_RETURN:
        """Transition the environment by 1 step. Optionally perform an action.

        Args:
            action (int, optional): The action to perform (1-indexed node to flip). Defaults to 0, meaning no action.

        Raises:
            Exception: When the action is outside the action space.

        Returns:
            GYM_STEP_RETURN: The typical Gymnasium environment 5-item Tuple.\
                 Consists of the resulting environment state, the associated reward, the termination / truncation status and additional info.
        """
        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        if action != 0:  # Action 0 is taking no action.
            self.PBN.flip(action)

        self.PBN.step()
        while not self.is_attracting_state(self.PBN.state):
            self.PBN.step()

        #self.step_no += 1

        observation = self.PBN.state
        reward, terminated, truncated = self._get_reward(observation, action)
        info = {"observation_idx": self._state_to_idx(observation)}

        return observation, reward, terminated, truncated, info

    def _get_reward(
        self, observation: STATE, action: int,
    ) -> Tuple[REWARD, TERMINATED, TRUNCATED]:
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED, TRUNCATED]: Tuple of the reward and the environment done status.
        """
        reward, terminated, truncated = 0, False, False
        observation_tuple = tuple(observation)

        if observation_tuple in self.target_nodes:
            reward += 20
            terminated = True
        else:
            if self.is_attracting_state(observation):
                reward -= 4
            else:
                raise ValueError

            if action != 0:
                reward -= 1
            else:
                reward -= 0

        # if self.step_no > 15:
        #     truncated = True

        return reward, terminated, truncated

    def reset(self, seed: int = None, options: dict = None) -> tuple[STATE, dict]:
        """Reset the environment. Initialise it to a random state."""
        if seed is not None:
            self._seed(seed)

        state = None
        if options is not None and "state" in options:
            print(f"options are {options}")
            state = options["state"]
        else:
            state = random.choice(tuple(self.attracting_states))

        attr = None
        while attr is None or len(attr) > 10:
            attr = random.choice(self.all_attractors)

        state = random.choice(tuple(attr))

        observation = self.PBN.reset(state)
        if tuple(observation) not in self.attracting_states:
            raise ValueError("state initial state should be an attractor")
        info = {"observation_idx": self._state_to_idx(observation)}
        self.step_no = 0
        return observation, info

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        no_cache = False

        if mode == "human":
            return self.PBN.state
        elif mode == "PBN":
            return self.PBN.print_PBN(no_cache)
        elif mode == "STG":
            return self.PBN.print_STG(no_cache)
        elif mode == "funcs":
            return self.PBN.print_functions()
        elif mode == "idx":
            return self._state_to_idx(self.PBN.state)
        elif mode == "float":
            return [float(x) for x in self.PBN.state]

    def _state_to_idx(self, state: STATE):
        return int(
            "".join([str(x) for x in np.array(state, dtype=np.int8).tolist()]), 2
        )

    def compute_attractors(self):
        print("Computing attractors...")
        STG = self.render(mode="STG")
        generator = nx.algorithms.components.attracting_components(STG)
        attractors = self._nx_attractors_to_tuples(list(generator))
        print(attractors)
        return attractors

    def _nx_attractors_to_tuples(self, attractors):
        return [
            set(
                [
                    tuple([int(x) for x in state.lstrip("[").rstrip("]").split()])
                    for state in list(attractor)
                ]
            )
            for attractor in attractors
        ]

    def clip(self, gene_i):
        self.PBN.clip(gene_i)

    def close(self):
        """Close out the environment and make sure everything is garbage collected."""
        del self.PBN
