import pickle
import random
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import List, Set, Tuple, Union
import pickle as pkl

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED

from .bittner import base, utils
from .bittner.base import findAttractors

from gym_PBN.utils.get_attractors_from_cabean import get_attractors


def state_equals(state1, state2):
    for x, y in zip(state1, state2):
        if x != y:
            return False
    return True


class PBNControlMultiEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "dict", "PBN", "STG", "idx", "float", "target"]
    }

    def __init__(
        self,
        graph: base.Graph,
        goal_config: dict,
        render_mode: str = None,
        render_no_cache: bool = False,
        name: str = None,
        end_episode_on_success: bool = False,
        horizon: int = 100,
        min_attractors=6,
        control_nodes=None,
    ):
        self.target = None
        self.graph = graph

        self.end_episode_on_success = end_episode_on_success

        print(f"setting {horizon}")
        self.horizon = horizon

        # Gym
        self.observation_space = MultiBinary(self.graph.N)
        # intervention nodes + no action
        print("\nhello\n")
        self.name = name
        self.render_mode = render_mode
        self.render_no_cache = render_no_cache

        # State
        self.n_steps = 0

        self.visited_states = defaultdict(int)

        self.all_attractors = []
        self.non_attractors = set()
        self.attracting_states = set()
        self.counter = 0

        # distribution for choosing starting end target attractors
        # initially uniform, will be to boost the frequency of hard cases
        self.probabilities = []
        self.initial_state = None
        self.target_state = None
        self.initial_state_id = -1
        self.target_state_id = -1

        self.target_attractor_id, self.state_attractor_id = -1, -1
        self.min_attractors = min_attractors

        if control_nodes is None:
            raise ValueError("Set of control nodes cannot be None")

        self.control_nodes = control_nodes
        self.action_space = MultiDiscrete(len(self.control_nodes))

    def step(self, actions, force=False, perturbation_prob=0.0):
        if not isinstance(actions, list):
            actions = actions.unique().tolist()

        # perturbation:
        if random.random() < perturbation_prob:
            state = list(self.graph.getState())
            flip = random.sample(range(self.graph.N), 3)

            for f in flip:
                state[f] = 1 - state[f]
                if f in actions:
                    actions.remove(f)

            self.graph.setState(state)

        self.n_steps += 1

        new_state = list(self.get_state())
        for node, value in zip(self.control_nodes, actions):
            new_state[node] = value

        self.graph.setState(new_state)

        observation = self.graph.getState()
        self.graph.step()

        step_count = 0
        returns_count = 0
        history = defaultdict(int)
        while not force and not self.is_attracting_state(observation):  # to liczy się na jednym cpu, i prawdobodobnie powoduje bottleneck w obliczeniach
            old_observation = observation
            observation = tuple(self.graph.step())

            if observation == old_observation:
                returns_count += 1

                if returns_count > 1000:
                    self.all_attractors.append([observation])
                    self.attracting_states.add(observation)
                    self.probabilities.append(0)
                    self.rework_probas()
                    with open(self.path, "wb+") as f:
                        pickle.dump(self.all_attractors, f)
                    break

            else:
                returns_count = 0

            step_count += 1
            history[observation] += 1

            if step_count > 10_000:
                states = sorted(history.items(), key=lambda kv: kv[1], reverse=True)
                new_attractors = [node for node, frequency in states if frequency > 1500]

                for s in new_attractors:
                    self.all_attractors.append([s])
                    self.attracting_states.add(s)
                    self.probabilities.append(0)

                self.rework_probas()
                step_count = 0
                history = defaultdict(int)
                with open(self.path, "wb+") as f:
                    pickle.dump(self.all_attractors, f)

        reward, terminated, truncated = self._get_reward(observation, actions)
        info = {
            # "observation_idx": self._state_to_idx(observation),
            # "observation_dict": observation,
        }

        return observation, reward, terminated, truncated, info

    def rework_probas(self, episode_len: int = 0):
        self.probabilities = [1/len(self.probabilities) for _ in self.probabilities]

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = getIDs()
            state = dict(zip(ids, state))
        return state

    def in_target(self, observation):
        for a_state in self.target:
            for i in range(len(a_state)):
                if a_state[i] == "*":
                    continue
                if a_state[i] != observation[i]:
                    break
            else:
                return True
        return False

    def _get_reward(self, observation: STATE, actions) -> Tuple[REWARD, TERMINATED, TRUNCATED]:

        if not isinstance(actions, list):
            actions = actions.tolist()
            actions = np.unique(actions)
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED, TRUNCATED]: Tuple of the reward and the environment done status.
        """
        reward, terminated = 0, False
        observation = tuple(observation)

        reward -= 1 * len(actions)

        if self.in_target(observation):
            reward += 100
            terminated = True

        truncated = self.n_steps == self.horizon
        return reward, terminated, truncated

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        if seed:
            self._seed(seed)

        self.target_attractor_id, self.state_attractor_id = np.random.choice(range(len(self.all_attractors)),
                                                                             size=2,
                                                                             replace=False)

        state_attractor = self.all_attractors[self.state_attractor_id]
        target_attractor = self.all_attractors[self.target_attractor_id]

        state = list(random.choice(state_attractor))
        target = list(random.choice(target_attractor))

        self.graph.setState(state)

        self.n_steps = 0
        observation = self.graph.getState()
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_dict": observation,
        }

        self.target = target_attractor
        return (tuple(state), tuple(target)), info

    def get_state(self):
        return np.array(self.graph.getState())

    def setTarget(self, target):
        self.target = target

    def render(self, mode=None):
        mode = self.render_mode if not mode else mode

        if mode == "human":
            return self.get_state()
        if mode == "dict":
            return self.graph.getState()
        elif mode == "PBN":
            return self.graph.printGraph()
        elif mode == "STG":
            return self.graph.genSTG()
        elif mode == "idx":
            return self._state_to_idx(self.graph.getState())
        elif mode == "float":
            return [float(x) for x in self.graph.getState()]
        elif mode == "target":
            state = self.graph.getState()
            return [state[node] for node in self.target_nodes]
        elif mode == "target_idx":
            target_state = self.render(mode="target")
            return self._state_to_idx(target_state)

    # AM: Added to hadle the problem of render() method not accepting the keyword argument 'mode'.
    def getTargetIdx(self):
        state = self.graph.getState()
        target_state = [state[node] for node in self.target_nodes]
        return self._state_to_idx(target_state)

    def _state_to_idx(self, state: STATE):
        if type(state) is dict:
            state = list(state.values())
        return int("".join([str(x) for x in state]), 2)

    def close(self):
        """Close out the environment and make sure everything is garbage collected."""
        del self.graph

    def statistical_attractors(self):
        state_log = defaultdict(int)

        self.setTarget([[0] * self.N])

        steps = 10**3
        min_attractors = self.min_attractors

        print(f"Calculating state statistics for N = {self.N}")
        print(f"running simulations. {steps} steps each")
        statistial_attractors = set()

        i = -1
        while len(statistial_attractors) < min_attractors:
            i += 1
            state_log = defaultdict(int)
            print(i)
            s = [random.randint(0, 1) for _ in range(self.N)]
            self.graph.setState(s)

            # warmup
            for _ in range(200):
                _ = self.step([], force=True)

            for j in range(steps):
                state = tuple(self.render())
                state_log[state] += 1
                _ = self.step([], force=True)

            states = sorted(state_log.items(), key=lambda kv: kv[1], reverse=True)

            statistial_attractors.update([node for node, frequency in states if frequency > 0.1 * steps])
            frequencies = sorted([frequency for node, frequency in states], reverse=True)[:10]
            print(f"(10%) calculating using {frequencies}. Got {len(statistial_attractors)}")

        print(f"got {statistial_attractors}")
        return statistial_attractors

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attracting_states

    def get_labels(self, state):
        nodes = self.graph.nodes
        return {node.ID: s for node, s in zip(nodes, state)}

    def get_next_state(self, state, actions):
        unlabeled_state = state
        state = self.get_labels(state)
        IDs = list(state.keys())

        if not isinstance(actions, list):
            actions = actions.unique().tolist()

        for action in actions:
            if action != 0:  # Action 0 is taking no action.
                ID = IDs[action - 1]
                state[ID] = 1 - state[ID]

        i = random.randint(0, len(state) - 1)
        new_state = state
        new_state[IDs[i]] = self.graph.nodes[i].step(state)
        unlabeled_new_state = tuple(new_state.values())

        step_count = 0
        returns_count = 0
        history = defaultdict(int)
        while not self.is_attracting_state(unlabeled_new_state):  # to liczy się na jednym cpu, i prawdobodobnie powoduje bottleneck w obliczeniach
            i = random.randint(0, len(self.graph.nodes) - 1)
            new_state = state
            new_state[IDs[i]] = self.graph.nodes[i].step(state)

            if unlabeled_state == unlabeled_new_state:
                returns_count += 1
            else:
                returns_count = 0

            unlabeled_state = unlabeled_new_state
            unlabeled_new_state = tuple(new_state.values())

            state = new_state

            if returns_count > 1_000:
                print(f"append {unlabeled_state} to attractor list")
                self.all_attractors.append([unlabeled_state])
                self.attracting_states.add(unlabeled_state)
                self.probabilities.append(0)
                self.rework_probas()
                return unlabeled_state

            step_count += 1
            history[unlabeled_state] += 1

            if step_count > 10_000:
                states = sorted(history.items(), key=lambda kv: kv[1], reverse=True)
                new_attractors = [node for node, frequency in states if frequency > 1500]

                print(len(new_attractors))
                for s in new_attractors:
                    print(s, history[s])
                    self.all_attractors.append([s])
                    self.attracting_states.add(s)
                    self.probabilities.append(0)

                self.rework_probas()
                step_count = 0

        return unlabeled_state


class BittnerControlGeneral(PBNControlMultiEnv):
    predictor_sets_path = Path(__file__).parent / "bittner" / "data"
    genedata = predictor_sets_path / "genedata.xls"

    includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    NAME = "Bittner-NONE"

    def __init__(
            self,
            N,
            render_mode: str = "human",
            render_no_cache: bool = False,
            name: str = None,
            horizon: int = 100,
            reward_config: dict = None,
            end_episode_on_success: bool = True,
            min_attractors=3,
            n_predictors=3
    ):
        if not name:
            name = self.NAME

        print(f"initing {name}")

        self.includeIDs = sorted(self.includeIDs)
        self.n_predictors = n_predictors
        self.N = N

        graph = utils.spawn(
            file=self.genedata,
            total_genes=self.N,
            include_ids=self.includeIDs,
            bin_method="median",
            n_predictors=n_predictors,
            k=3,
            predictor_sets_path=self.predictor_sets_path,
        )

        goal_config = {
            "target_nodes": [234237, 324901, 759948, 25485, 266361, 108208, 130057],
            "intervene_on": [234237, 324901, 759948, 25485, 266361, 108208, 130057],
            "target_node_values": ((1, 1, 1, 1, 1, 1, 0),),
            "undesired_node_values": tuple(),
            "horizon": horizon,
        }
        super().__init__(
            graph,
            goal_config,
            render_mode,
            render_no_cache,
            name,
            end_episode_on_success,
        )

        self.horizon = horizon
        self.min_attractors = min_attractors
        print("Single episode horizon is ", self.horizon)

        self.path = f"attractors/{self.N}_{self.n_predictors}_attractors.pkl"
        try:
            print(f"try to load: \n{self.path}")
            with open(self.path, "rb") as f:
                attractors = pickle.load(f)
                self.all_attractors = attractors
        except FileNotFoundError:
            self.all_attractors = [[s] for s in self.statistical_attractors()]
            with open(self.path, "wb+") as f:
                pickle.dump(self.all_attractors, f)

        for a in self.all_attractors:
            self.attracting_states.add(a[0])

        self.attractor_count = len(self.all_attractors)
        self.probabilities = [1 / self.attractor_count] * self.attractor_count

        print(self.all_attractors)
