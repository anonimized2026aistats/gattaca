import pickle
import random
from collections import defaultdict
from typing import List, Tuple
import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium.spaces import MultiBinary, MultiDiscrete

from bang.core.pbn.utils.state_printing import convert_to_binary_representation
from gym_PBN.envs.pbn_env import get_model_io
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED

import bang
from bang.core.attractors.monte_carlo.merge_attractors import merge_attractors


def read_ispl_node_names(path):
    genes = []

    with open(path, "r") as env_file:
        for line in env_file:
            line = line.split()

            if len(line) == 0:
                continue

            if line[0] == "Vars:":
                while True:
                    line = next(env_file).split()

                    if line[0] == "end":
                        break

                    if line[0][-1] == ":":
                        genes.append(line[0][:-1])
                    else:
                        genes.append(line[0])

                break

    return genes


def resolve_node_indices(genes, node_names):
    indices = []
    for node_name in node_names:
        candidates = [node_name]
        if not node_name.startswith("v_"):
            candidates.append(f"v_{node_name}")

        for candidate in candidates:
            if candidate in genes:
                indices.append(genes.index(candidate))
                break
        else:
            raise ValueError(f"Node {node_name!r} not found in ISPL node list")

    return indices


def packed_state_to_int(state):
    packed_state = np.atleast_1d(state)
    value = 0

    for i, chunk in enumerate(packed_state):
        value |= int(chunk) << (32 * i)

    return value


def packed_state_to_bools(state, n_nodes):
    value = packed_state_to_int(state)
    return [bool((value >> i) & 1) for i in range(n_nodes)]


def history_to_trajectories(history):
    return [
        [packed_state_to_int(history[step, trajectory]) for step in range(history.shape[0])]
        for trajectory in range(history.shape[1])
    ]


class BangGraph:
    def __init__(self, pbn):
        self.pbn = pbn

    def get_adj_list(self):
        top_nodes = []
        bot_nodes = []
        function_index = 0

        for node_index, function_count in enumerate(self.pbn.n_functions):
            done = set()
            top_nodes.append(node_index)
            bot_nodes.append(node_index)

            for _ in range(function_count):
                for parent_index in self.pbn.parent_variable_indices[function_index]:
                    if parent_index not in done:
                        done.add(parent_index)
                        top_nodes.append(node_index)
                        bot_nodes.append(parent_index)
                function_index += 1

        return [top_nodes, bot_nodes]


class PBNBangEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "dict", "PBN", "STG", "idx", "float", "target"]
    }

    def __init__(
        self,
        file: str,
        input_nodes: List[int] = None,
        initial_values: list[int] = None,
        forbidden_nodes: list[int] = None,
        target_nodes: List[int] = None,
        target_values: List[int] = None,
        format: str = "sbml",
        goal_config: dict = None,
        model_name=None,
        render_mode: str = None,
        render_no_cache: bool = False,
        name: str = None,
        end_episode_on_success: bool = False,
        horizon: int = 100,
        n_parallel=2,
        pasip_history_len=10,
        pasip_size=10000,
        settle_steps=10_000,
        update_type="asynchronous_one_random",
    ):
        self.target = None
        self.end_episode_on_success = end_episode_on_success

        if model_name is not None and any(
            item is None for item in [input_nodes, initial_values, target_nodes, target_values]
        ):
            genes = read_ispl_node_names(file)
            input_node_names, initial_values, target_node_names, target_values = get_model_io(model_name)
            input_nodes = resolve_node_indices(genes, input_node_names)
            target_nodes = resolve_node_indices(genes, target_node_names)

        input_nodes = [] if input_nodes is None else input_nodes
        initial_values = [] if initial_values is None else initial_values
        forbidden_nodes = [] if forbidden_nodes is None else forbidden_nodes
        target_nodes = [] if target_nodes is None else target_nodes
        target_values = [] if target_values is None else target_values

        print(f"Loading BANG PBN from {file} ({format})...", flush=True)
        self.pbn = bang.load_from_file(file, format=format, n_parallel=n_parallel)
        self.pbn.update_type = update_type
        print(f"BANG update type: {self.pbn.update_type}", flush=True)
        print("Detecting initial attractors with BANG Monte Carlo...", flush=True)
        attractors = self.pbn.monte_carlo_detect_attractors(pasip_history_len, pasip_size, repr='bool')
        print(f"Detected {len(attractors)} attractor components.", flush=True)

        self.graph = BangGraph(self.pbn)
        self.observation_space = MultiBinary(self.pbn.n_nodes)
        self.action_space = MultiDiscrete(self.pbn.n_nodes + 1)
        self.blocks = None
        self.all_attractors = attractors[0]
        print(attractors, flush=True)

        print(f"setting {horizon}", flush=True)
        self.horizon = horizon
        self.settle_steps = settle_steps

        # Gym
        print("\nhello\n", flush=True)
        self.name = name
        self.render_mode = render_mode
        self.render_no_cache = render_no_cache

        # State
        self.n_steps = 0

        self.attracting_states = set()
        self.counter = 0

        self.initial_state_id = -1
        self.target_state_id = -1

        self.target_attractor_id, self.state_attractor_id = -1, -1
        self.forbidden_actions = self.forbidden_actions = list(set(input_nodes).union(forbidden_nodes))
        self.input_nodes = input_nodes

        self.target_nodes = target_nodes
        self.target_values = target_values

        self.attractor_set = {tuple(a) for a in self.all_attractors}
        self.divided_attractors = [a for a in self.all_attractors if
                                   not (self.in_target(a))]
        self.target_attractors = [a for a in self.all_attractors if
                                  self.in_target(a)]

        if len(self.divided_attractors) == 0:
            print("THERE IS NO VALID SOURCE ATTRACTOR")
        if len(self.target_attractors) == 0:
            print("THERE IS NO VALID TARGET ATTRACTOR")

    def _seed(self, seed: int = None):
        np.random.seed(seed)
        random.seed(seed)

    def get_id(self, state):
        for i, attractor in enumerate(self.all_attractors):
            if state == attractor[0]:
                return i

        raise ValueError

    def set_input_nodes(self, input_nodes):
        self.input_nodes = input_nodes

    def step(self, actions, force=False, perturbation_prob=0.0):
        if not isinstance(actions, list):
            actions = actions.unique().tolist()

        self.n_steps += 1

        actions = [
            int(action.item()) if hasattr(action, "item") else int(action)
            for action in actions
        ]
        actions = [action - 1 for action in actions if action > 0 and action - 1 not in self.forbidden_actions]
        bang_actions = np.array(actions, dtype=np.uint32) if actions else None

        self.pbn.simple_steps(n_steps=1, actions=bang_actions)
        observation = packed_state_to_bools(self.pbn._latest_state[0], self.pbn.n_nodes)

        if not force and not self.is_attracting_state(observation):
            self.pbn.simple_steps(n_steps=self.settle_steps)
            observation = packed_state_to_bools(self.pbn._latest_state[0], self.pbn.n_nodes)

            attractors = merge_attractors(history_to_trajectories(self.pbn.history[1:]))
            attractors = convert_to_binary_representation(attractors, self.pbn._n)[0]

            # type II pseudo-attractor
            for a in attractors:
                if tuple(a) not in self.attracting_states:
                    self.all_attractors.append(a)
                    self.attracting_states.add(tuple(a))

                # with open(self.path, "wb+") as f:
                #     pickle.dump(self.all_attractors, f)

            if tuple(observation) not in self.attracting_states:
                self.all_attractors.append(observation)
                self.attracting_states.add(tuple(observation))
                self.attractor_set.add(tuple(observation))

        reward, terminated, truncated = self._get_reward(observation, actions)

        return observation, reward, terminated, truncated, {}

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = getIDs()
            state = dict(zip(ids, state))
        return state

    def in_target(self, observation):
        for i in range(len(self.target_values)):
            if observation[self.target_nodes[i]] != self.target_values[i]:
                return False
        return True

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
        reward, terminated = 41, False
        observation = tuple(observation)

        reward -= 1 * len(actions)

        if len(actions) > 10:
            reward -= 10 * (len(actions) - 10)

        if self.in_target(observation):
            reward += 1000
            terminated = True

        truncated = self.n_steps == self.horizon
        return reward, terminated, truncated

    def calculate_attractors(self, n_paths, path_len, trajectory_length=1000):
        if n_paths < 2:
            n_paths = 2

        self.pbn._n_parallel = n_paths
        attractors = self.pbn.monte_carlo_detect_attractors(trajectory_length=trajectory_length, attractor_length=path_len)
        return attractors

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        if seed:
            self._seed(seed)

        state = target = None

        if len(self.divided_attractors) > 0:
            state = random.choice(self.divided_attractors)
        else:
            state = self.target_attractors[0]

        self.pbn.set_states([state])

        self.n_steps = 0
        observation = [int(x) for x in state]
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_di ct": observation,
        }

        self.target = None
        # print(state)
        # print(self.target_attractors)
        # print(len(self.divided_attractors))
        # print(len(self.target_attractors))
        return tuple(state), info

    def get_state(self):
        return np.array(self.graph.getState())

    def setTarget(self, target):
        self.target = target

    def render(self, mode=None):
        mode = self.render_mode if not mode else mode

        return self.pbn.history_bool[-1][0]

    # AM: Added to hadle the problem of render() method not accepting the keyword argument 'mode'.
    def getTargetIdx(self):
        state = self.graph.getState()
        target_state = [state[node] for node in self.target_nodes]
        return self._state_to_idx(target_state)

    def _state_to_idx(self, state: STATE):
        if type(state) is dict:
            state = list(state.values())
        return int("".join([str(x) for x in state]), 2)

    def compute_attractors(self):
        print("Computing attractors...")
        STG = self.render(mode="STG")
        generator = nx.algorithms.components.attracting_components(STG)
        return self._nx_attractors_to_tuples(list(generator))

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

    def close(self):
        """Close out the environment and make sure everything is garbage collected."""
        pass

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
                    # print(s, history[s])
                    self.all_attractors.append([s])
                    self.attracting_states.add(s)
                    self.probabilities.append(0)

                self.rework_probas()
                step_count = 0

        return unlabeled_state
