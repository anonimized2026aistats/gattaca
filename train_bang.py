import argparse
import pickle
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from gym_PBN.envs.pbn_bang_env import (
    PBNBangEnv,
    packed_state_to_bools,
    read_ispl_node_names,
    resolve_node_indices,
)
from gym_PBN.envs.pbn_env import get_model_io

from gattaca_model import GATTACA
from gattaca_model.utils import AgentConfig

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


def normalize_assa_expression(expression):
    expression = expression.strip().rstrip(";")
    expression = expression.replace("~", "!")
    expression = re.sub(r"\bnot\b", "!", expression)
    expression = re.sub(r"\band\b", "&", expression)
    expression = re.sub(r"\bor\b", "|", expression)
    expression = re.sub(r"\btrue\b", "True", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bfalse\b", "False", expression, flags=re.IGNORECASE)
    return expression


def parse_ispl_for_assa(path):
    genes = []
    rules = {}
    true_rule_pattern = re.compile(r"^\s*([A-Za-z0-9_]+)=true\s+if\s+(.+)=true;?\s*$")

    with open(path, "r") as env_file:
        in_vars = False
        in_evolution = False
        for line in env_file:
            stripped = line.strip()

            if stripped == "Vars:":
                in_vars = True
                continue
            if in_vars:
                if stripped.startswith("end"):
                    in_vars = False
                    continue
                parts = stripped.split()
                if parts:
                    genes.append(parts[0].rstrip(":"))
                continue

            if stripped == "Evolution:":
                in_evolution = True
                continue
            if in_evolution and stripped.startswith("end"):
                break
            if not in_evolution:
                continue

            match = true_rule_pattern.match(stripped)
            if match is None:
                continue

            gene, expression = match.groups()
            rules[gene] = normalize_assa_expression(expression)

    missing = [gene for gene in genes if gene not in rules]
    if missing:
        raise ValueError(f"Missing ISPL update rules for {missing}")

    return genes, rules


def parse_bnet_for_assa(path):
    genes = []
    rules = {}

    with open(path, "r") as env_file:
        for line in env_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.lower() == "targets,factors":
                continue

            target, expression = [part.strip() for part in stripped.split(",", maxsplit=1)]
            genes.append(target)
            rules[target] = normalize_assa_expression(expression)

    return genes, rules


def write_assa_file(genes, rules, fixed_values=None):
    fixed_values = {} if fixed_values is None else fixed_values
    assa_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pbn",
        prefix="gattaca_bang_",
        delete=False,
    )

    with assa_file:
        assa_file.write("type=synchronous\n")
        assa_file.write(f"n={len(genes)}\n")
        assa_file.write("perturbation=0.0\n")
        assa_file.write("nodeNames\n")
        for gene in genes:
            assa_file.write(f"{gene}\n")
        assa_file.write("endNodeNames\n")

        for gene in genes:
            if fixed_values.get(gene) is True:
                rule = f"({gene} | !{gene})"
            elif fixed_values.get(gene) is False:
                rule = f"({gene} & !{gene})"
            else:
                rule = rules.get(gene, gene)
            assa_file.write(f"node {gene}\n")
            assa_file.write(f"1.0: {rule}\n")
            assa_file.write("endNode\n")

        assa_file.write("npNode\n")
        assa_file.write("endNpNode\n")

    return assa_file.name


def patch_bang_merge_attractors(threshold=0.2, include_final_states=False):
    def normalize_history_state(state):
        state = np.asarray(state, dtype=bool)
        if state.ndim == 2:
            node_count = state.shape[1]
            chunks = []
            for chunk_index, chunk in enumerate(state):
                start = chunk_index * 32
                if start >= node_count:
                    break
                chunks.extend(chunk[: min(32, node_count - start)].tolist())
            return tuple(chunks)

        return tuple(state.reshape(-1).tolist())

    def merge_attractors(data, threshold=threshold):
        attractors = set()
        trajectory_len = len(data)
        trajectory_count = len(data[0])

        for trajectory in range(trajectory_count):
            histogram = defaultdict(int)
            for step in range(trajectory_len):
                histogram[normalize_history_state(data[step][trajectory])] += 1

            attractors.update(
                node
                for node, count in histogram.items()
                if count >= threshold * trajectory_len
            )
            if include_final_states:
                attractors.add(normalize_history_state(data[-1][trajectory]))

        return [[np.array(attractor) for attractor in attractors]]

    import bang.core.attractors.monte_carlo.merge_attractors as merge_module
    import bang.core.attractors.monte_carlo.monte_carlo as monte_carlo_module

    merge_module.merge_attractors = merge_attractors
    monte_carlo_module.merge_attractors = merge_attractors


def prepare_bang_source(path, bang_format, model_name):
    if bang_format == "ispl":
        genes, rules = parse_ispl_for_assa(path)
    elif bang_format == "bnet":
        genes, rules = parse_bnet_for_assa(path)
    else:
        return path, bang_format, None, None, None, None, None

    input_node_names, initial_values, target_node_names, target_values = get_model_io(model_name)
    input_nodes = resolve_node_indices(genes, input_node_names)
    target_nodes = resolve_node_indices(genes, target_node_names)
    fixed_values = {
        node_name if node_name in genes else f"v_{node_name}": bool(value)
        for node_name, value in zip(input_node_names, initial_values)
    }

    return (
        write_assa_file(genes, rules, fixed_values=fixed_values),
        "assa",
        input_nodes,
        initial_values,
        target_nodes,
        target_values,
        genes,
    )


class TrainingPBNBangEnv(PBNBangEnv):
    is_vectorized = True

    def __init__(
        self,
        *args,
        bang_device="cuda",
        attractor_device="cpu",
        update_type="asynchronous_one_random",
        train_parallel=None,
        attractor_cache=None,
        force_attractor_refresh=False,
        node_names=None,
        input_values=None,
        use_statistical_attractors=True,
        statistical_warmup=1000,
        statistical_steps=10_000,
        statistical_min_attractors=3,
        statistical_trajectories=1,
        statistical_validation_steps=1000,
        statistical_require_validation=False,
        runtime_attractor_discovery=False,
        runtime_attractor_threshold=0.15,
        runtime_attractor_trajectories=1,
        runtime_attractor_max_additions_per_step=16,
        runtime_attractor_max_attractors=0,
        **kwargs,
    ):
        super().__init__(*args, update_type=update_type, **kwargs)
        monte_carlo_attractors = [list(map(bool, attractor)) for attractor in self.all_attractors]
        self.bang_device = bang_device
        self.attractor_device = attractor_device
        self.num_envs = int(train_parallel or self.pbn._n_parallel)
        self.node_names = list(node_names) if node_names is not None else None
        self.input_values = [] if input_values is None else [bool(value) for value in input_values]
        self.attractor_cache = Path(attractor_cache) if attractor_cache is not None else None
        self.runtime_attractor_discovery = runtime_attractor_discovery
        self.runtime_attractor_threshold = runtime_attractor_threshold
        self.runtime_attractor_trajectories = runtime_attractor_trajectories
        self.runtime_attractor_max_additions_per_step = runtime_attractor_max_additions_per_step
        self.runtime_attractor_max_attractors = runtime_attractor_max_attractors
        if self.runtime_attractor_discovery:
            input_summary = [
                f"{self.node_names[node] if self.node_names is not None else node}={int(value)}"
                for node, value in zip(self.input_nodes, self.input_values)
            ]
            print(
                "Runtime PASIP-II attractor discovery enabled "
                f"(inputs: {', '.join(input_summary)}, "
                f"trajectories: {self.runtime_attractor_trajectories}, "
                f"threshold: {self.runtime_attractor_threshold})",
                flush=True,
            )
        if self.bang_device == "cuda" and self.num_envs % 32 != 0:
            raise ValueError("--train-parallel must be a multiple of 32 when using BANG CUDA")
        if self.attractor_cache is not None and self.attractor_cache.exists() and not force_attractor_refresh:
            self._sync_attractor_cache(force_attractor_refresh)
        elif use_statistical_attractors:
            self.all_attractors = self._statistical_attractors(
                warmup_steps=statistical_warmup,
                sample_steps=statistical_steps,
                min_attractors=statistical_min_attractors,
                trajectory_count=statistical_trajectories,
                validation_steps=statistical_validation_steps,
                require_validation=statistical_require_validation,
            )
            if len(self.all_attractors) == 0:
                print("BANG statistical validation found no attractors; keeping Monte Carlo attractors", flush=True)
                self.all_attractors = monte_carlo_attractors
            self._refresh_attractor_splits()
            if len(self.target_attractors) == 0:
                self._try_add_target_attractor(validation_steps=statistical_validation_steps)
        elif self.attractor_cache is not None:
            self._sync_attractor_cache(force_attractor_refresh)
        self.attracting_states = {tuple(attractor) for attractor in self.all_attractors}
        self.attractor_set = set(self.attracting_states)
        self.forbidden_actions = sorted(set(self.forbidden_actions).union(self.target_nodes))
        self.pbn.save_history = False
        self.n_steps = np.zeros(self.num_envs, dtype=np.int32)

    def _refresh_attractor_splits(self):
        self.attractor_set = {tuple(attractor) for attractor in self.all_attractors}
        self.attracting_states = set(self.attractor_set)
        self.divided_attractors = [
            attractor for attractor in self.all_attractors
            if not self.in_target(attractor)
        ]
        self.target_attractors = [
            attractor for attractor in self.all_attractors
            if self.in_target(attractor)
        ]

    def _random_states_with_fixed_inputs(self, count):
        states = np.random.randint(0, 2, size=(count, self.pbn.n_nodes)).astype(bool)
        for node, value in zip(self.input_nodes, self.input_values):
            states[:, node] = value
        return states.tolist()

    def _current_states(self):
        return [
            tuple(packed_state_to_bools(state, self.pbn.n_nodes))
            for state in self.pbn._latest_state
        ]

    def _inputs_match_initial_values(self, state):
        return all(
            bool(state[node]) == value
            for node, value in zip(self.input_nodes, self.input_values)
        )

    def _add_runtime_attractor(self, state):
        state = tuple(bool(value) for value in state)
        if state in self.attracting_states:
            return False
        if not self._inputs_match_initial_values(state):
            return False
        if (
            self.runtime_attractor_max_attractors > 0
            and len(self.all_attractors) >= self.runtime_attractor_max_attractors
        ):
            return False

        self.all_attractors.append([bool(value) for value in state])
        self._refresh_attractor_splits()
        return True

    def _discover_runtime_attractors_from_history(self):
        if not self.runtime_attractor_discovery:
            return 0

        history = self.pbn.history
        if history.shape[0] == 0:
            return 0

        trajectory_len = history.shape[0]
        trajectory_count = min(self.runtime_attractor_trajectories, history.shape[1])
        threshold_count = self.runtime_attractor_threshold * trajectory_len
        added = 0

        for trajectory in range(trajectory_count):
            histogram = defaultdict(int)
            for step in range(trajectory_len):
                state = tuple(packed_state_to_bools(history[step, trajectory], self.pbn.n_nodes))
                histogram[state] += 1

            for state, count in sorted(histogram.items(), key=lambda item: item[1], reverse=True):
                if count <= threshold_count:
                    break
                if self._add_runtime_attractor(state):
                    added += 1
                    if (
                        self.runtime_attractor_max_additions_per_step > 0
                        and added >= self.runtime_attractor_max_additions_per_step
                    ):
                        print(f"Added {added} runtime attractors from PASIP-II", flush=True)
                        self.save_attractor_cache()
                        return added

        if added:
            print(f"Added {added} runtime attractors from PASIP-II", flush=True)
            self.save_attractor_cache()
        return added

    def _validate_attractor_candidate(self, candidate, validation_steps):
        candidate = tuple(bool(value) for value in candidate)
        self.pbn.set_states([list(candidate)], reset_history=True)
        for _ in range(validation_steps):
            self.pbn.simple_steps(n_steps=1, device=self.attractor_device)
            if self._current_states()[0] == candidate:
                return True
        return False

    def _statistical_attractors(
        self,
        warmup_steps,
        sample_steps,
        min_attractors,
        trajectory_count,
        validation_steps,
        require_validation,
    ):
        print("Calculating BANG old-style statistical attractors...", flush=True)
        previous_save_history = self.pbn.save_history
        self.pbn.save_history = False

        candidates = set()
        runs = 0
        max_runs = max(10, min_attractors * 10 + 1)

        while len(candidates) < min_attractors and runs < max_runs:
            runs += 1
            states = self._random_states_with_fixed_inputs(self.pbn._n_parallel)
            self.pbn.set_states(states, reset_history=True)
            self.pbn.simple_steps(n_steps=warmup_steps, device=self.attractor_device)

            histograms = [defaultdict(int) for _ in range(self.pbn._n_parallel)]
            for _ in range(sample_steps):
                for index, state in enumerate(self._current_states()):
                    histograms[index][state] += 1
                self.pbn.simple_steps(n_steps=1, device=self.attractor_device)

            new_candidates = set()
            threshold_count = 0.2 * sample_steps
            for histogram in histograms[: min(trajectory_count, len(histograms))]:
                new_candidates.update(
                    state for state, count in histogram.items()
                    if count > threshold_count
                )

            candidates.update(new_candidates)
            top_counts = sorted(
                [max(histogram.values()) for histogram in histograms if len(histogram) > 0],
                reverse=True,
            )[:10]
            print(
                f"({len(candidates)}, {runs}) BANG statistical candidates; top counts {top_counts}",
                flush=True,
            )

        validated = []
        for candidate in candidates:
            if self._validate_attractor_candidate(candidate, validation_steps):
                validated.append([bool(value) for value in candidate])

        self.pbn.save_history = previous_save_history
        print(
            f"Validated {len(validated)} BANG statistical attractors "
            f"from {len(candidates)} candidates",
            flush=True,
        )
        if require_validation:
            return validated
        return [[bool(value) for value in candidate] for candidate in candidates]

    def _try_add_target_attractor(self, validation_steps):
        print("Trying old-env-style target attractor fallback...", flush=True)
        trajectory_count = max(32, self.pbn._n_parallel)

        for attempt in range(10):
            seed_states = self._random_states_with_fixed_inputs(trajectory_count)
            if attempt == 0:
                seed_states[0] = [True] * self.pbn.n_nodes
                seed_states[1] = [False] * self.pbn.n_nodes

            for state in seed_states:
                for node, value in zip(self.input_nodes, self.input_values):
                    state[node] = bool(value)
                for node, value in zip(self.target_nodes, self.target_values):
                    state[node] = bool(value)

            self.pbn.set_states(seed_states, reset_history=True)
            self.pbn.simple_steps(n_steps=self.settle_steps, device=self.attractor_device)
            for candidate in self._current_states():
                if not self.in_target(candidate):
                    continue
                if tuple(candidate) in self.attractor_set:
                    return

                # The old environment accepted a target state reached after settling
                # as a target pseudo-attractor without singleton validation.
                self.all_attractors.append([bool(value) for value in candidate])
                self._refresh_attractor_splits()
                print("Added target attractor from fallback", flush=True)
                return

        print("Target fallback did not reach the requested target phenotype", flush=True)

    def _sync_attractor_cache(self, force_attractor_refresh):
        if self.attractor_cache.exists() and not force_attractor_refresh:
            with open(self.attractor_cache, "rb") as cache_file:
                loaded = pickle.load(cache_file)

            self.all_attractors = [
                [bool(value) for value in (attractor[0] if isinstance(attractor, list) else attractor)]
                for attractor in loaded
            ]
            self._refresh_attractor_splits()
            print(
                f"Loaded {len(self.all_attractors)} BANG attractors from {self.attractor_cache} "
                f"({len(self.divided_attractors)} source, {len(self.target_attractors)} target)"
            )
            return

        self.save_attractor_cache()

    def save_attractor_cache(self):
        if self.attractor_cache is None:
            return

        self._refresh_attractor_splits()
        self.attractor_cache.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            [tuple(int(bool(value)) for value in attractor)]
            for attractor in self.all_attractors
        ]
        with open(self.attractor_cache, "wb") as cache_file:
            pickle.dump(serializable, cache_file)
        print(
            f"Saved {len(self.all_attractors)} BANG attractors to {self.attractor_cache} "
            f"({len(self.divided_attractors)} source, {len(self.target_attractors)} target)"
        )

    def _sample_source_state(self):
        if len(self.divided_attractors) > 0:
            return random.choice(self.divided_attractors)
        if len(self.target_attractors) > 0:
            return random.choice(self.target_attractors)
        return random.choice(self.all_attractors)

    def reset(self, seed=None, options=None, indices=None):
        if seed is not None:
            self._seed(seed)

        if indices is None:
            states = [self._sample_source_state() for _ in range(self.num_envs)]
            self.n_steps = np.zeros(self.num_envs, dtype=np.int32)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            states = [
                packed_state_to_bools(state, self.pbn.n_nodes)
                for state in self.pbn._latest_state
            ]
            for index in indices:
                states[index] = self._sample_source_state()
                self.n_steps[index] = 0

        states = [list(map(bool, state)) for state in states]
        self.pbn.set_states(states, reset_history=True)
        observations = np.asarray(states, dtype=bool)
        return observations, {}

    def _normalize_action_rows(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, axis=0)

        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} action rows, got {actions.shape[0]}")

        rows = []
        for row in actions:
            action_row = []
            for action in np.unique(row):
                action = int(action)
                node_index = action - 1
                if action > 0 and node_index not in self.forbidden_actions:
                    action_row.append(node_index)
            rows.append(action_row)
        return rows

    def _perturb_state_rows(self, action_rows):
        latest_state = self.pbn._latest_state.copy()

        for row_index, row_actions in enumerate(action_rows):
            for action in row_actions:
                state_index = action // 32
                bit_index = action % 32
                latest_state[row_index, state_index] ^= np.uint32(1 << bit_index)

        self.pbn._latest_state = latest_state

    def _observations(self):
        return np.asarray(
            [
                packed_state_to_bools(state, self.pbn.n_nodes)
                for state in self.pbn._latest_state
            ],
            dtype=bool,
        )

    def _get_reward_batch(self, observations, action_rows):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)

        for index, observation in enumerate(observations):
            rewards[index] -= len(action_rows[index])

            if self.in_target(observation):
                rewards[index] += 1000
                terminated[index] = True

        truncated = self.n_steps >= self.horizon
        return rewards, terminated, truncated

    def step(self, actions, force=False, perturbation_prob=0.0):
        action_rows = self._normalize_action_rows(actions)
        self.n_steps += 1

        self._perturb_state_rows(action_rows)
        self.pbn.simple_steps(n_steps=1, device=self.bang_device)
        observations = self._observations()

        if not force and any(not self.is_attracting_state(observation) for observation in observations):
            previous_save_history = self.pbn.save_history
            self.pbn.save_history = self.runtime_attractor_discovery
            try:
                self.pbn.simple_steps(n_steps=self.settle_steps, device=self.bang_device)
                observations = self._observations()
                self._discover_runtime_attractors_from_history()
            finally:
                self.pbn.save_history = previous_save_history

        reward, terminated, truncated = self._get_reward_batch(observations, action_rows)

        return observations, reward, terminated, truncated, {}


def build_config(args):
    config = AgentConfig()
    config.batch_size = args.batch_size
    config.memory_size = args.memory_size
    config.learning_starts = max(args.learning_starts, args.batch_size)
    config.time_steps = args.time_steps
    config.bins = args.bins

    if args.no_cuda:
        config.device = torch.device("cpu")
        print("Training on cpu")

    return config


def get_latest_checkpoint(checkpoint_path):
    files = list(checkpoint_path.glob("*.pt"))
    if len(files) > 0:
        return max(files, key=lambda x: x.stat().st_ctime)
    return None


def debug_attractors(env, limit=20, save_path=None):
    print("Attractor debug")
    print(f"node_count: {env.pbn.n_nodes}")
    if env.node_names is not None:
        print("node order:")
        for index, name in enumerate(env.node_names[:limit]):
            print(f"  {index}: {name}")
    print(f"input_nodes: {env.input_nodes}")
    if env.node_names is not None:
        print(f"input_node_names: {[env.node_names[node] for node in env.input_nodes]}")
    print(f"target_nodes: {env.target_nodes}")
    if env.node_names is not None:
        print(f"target_node_names: {[env.node_names[node] for node in env.target_nodes]}")
    print(f"target_values: {env.target_values}")
    print(f"all_attractors: {len(env.all_attractors)}")
    print(f"source_attractors: {len(env.divided_attractors)}")
    print(f"target_attractors: {len(env.target_attractors)}")

    target_projection_counts = defaultdict(int)
    for attractor in env.all_attractors:
        projection = tuple(bool(attractor[node]) for node in env.target_nodes)
        target_projection_counts[projection] += 1

    print("target projections:")
    for projection, count in sorted(target_projection_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {projection}: {count}")

    for index, attractor in enumerate(env.all_attractors[:limit]):
        projection = [int(bool(attractor[node])) for node in env.target_nodes]
        print(
            f"attractor[{index}] target={projection} "
            f"is_target={env.in_target(attractor)} "
            f"state_prefix={[int(bool(value)) for value in attractor[:16]]}"
        )

    if save_path is not None:
        import json

        payload = {
            "target_nodes": env.target_nodes,
            "target_values": env.target_values,
            "attractors": [[int(bool(value)) for value in attractor] for attractor in env.all_attractors],
        }
        with open(save_path, "w") as debug_file:
            json.dump(payload, debug_file, indent=2)
        print(f"saved attractor debug to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train GATTACA with a BANG-backed PBN environment.")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--exp-name", type=str, default="bang_ddqn")
    parser.add_argument("--env", type=str, default="bang")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--assa-file", type=str, required=True)
    parser.add_argument("--bang-format", choices=["ispl", "bnet", "assa", "sbml"], default="ispl")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-size", type=int, default=10**5)
    parser.add_argument("--learning-starts", type=int, default=278)
    parser.add_argument("--time-steps", type=int, default=10_000_000)
    parser.add_argument("--bins", type=int, default=5)

    parser.add_argument("--bang-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--attractor-device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument(
        "--bang-update-type",
        choices=["asynchronous_one_random", "asynchronous_random_order", "synchronous"],
        default="asynchronous_one_random",
    )
    parser.add_argument("--n-parallel", type=int, default=512)
    parser.add_argument("--train-parallel", type=int, default=None)
    parser.add_argument("--settle-steps", type=int, default=10_000)
    parser.add_argument("--pasip-history-len", type=int, default=1000)
    parser.add_argument("--pasip-size", type=int, default=100)
    parser.add_argument("--attractor-threshold", type=float, default=0.2)
    parser.add_argument("--attractor-include-final-states", action="store_true", default=False)
    parser.add_argument("--no-statistical-attractors", action="store_true", default=False)
    parser.add_argument("--statistical-warmup", type=int, default=1000)
    parser.add_argument("--statistical-steps", type=int, default=10_000)
    parser.add_argument("--statistical-min-attractors", type=int, default=3)
    parser.add_argument("--statistical-trajectories", type=int, default=1)
    parser.add_argument("--statistical-validation-steps", type=int, default=1000)
    parser.add_argument("--statistical-require-validation", action="store_true", default=False)
    parser.add_argument("--runtime-attractor-discovery", action="store_true", default=False)
    parser.add_argument("--runtime-attractor-threshold", type=float, default=0.15)
    parser.add_argument("--runtime-attractor-trajectories", type=int, default=1)
    parser.add_argument("--runtime-attractor-max-additions-per-step", type=int, default=16)
    parser.add_argument("--runtime-attractor-max-attractors", type=int, default=0)
    parser.add_argument("--attractor-cache", type=str, default=None)
    parser.add_argument("--force-attractor-refresh", action="store_true", default=False)
    parser.add_argument("--save-bang-history", action="store_true", default=False)
    parser.add_argument("--debug-attractors", action="store_true", default=False)
    parser.add_argument("--debug-attractors-save", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()

    if args.bang_device == "cuda" and args.n_parallel % 32 != 0:
        raise ValueError("--n-parallel must be a multiple of 32 when using BANG CUDA")
    if args.bang_device == "cuda" and args.train_parallel is not None and args.train_parallel % 32 != 0:
        raise ValueError("--train-parallel must be a multiple of 32 when using BANG CUDA")

    patch_bang_merge_attractors(
        threshold=args.attractor_threshold,
        include_final_states=args.attractor_include_final_states,
    )

    bang_file, bang_format, input_nodes, initial_values, target_nodes, target_values, node_names = prepare_bang_source(
        args.assa_file,
        args.bang_format,
        args.model_name,
    )

    try:
        env = TrainingPBNBangEnv(
            file=bang_file,
            format=bang_format,
            model_name=None,
            input_nodes=input_nodes,
            target_nodes=target_nodes,
            target_values=target_values,
            name=args.model_name,
            horizon=100,
            n_parallel=args.n_parallel,
            pasip_history_len=args.pasip_history_len,
            pasip_size=args.pasip_size,
            settle_steps=args.settle_steps,
            bang_device=args.bang_device,
            attractor_device=args.attractor_device,
            update_type=args.bang_update_type,
            train_parallel=args.train_parallel,
            attractor_cache=args.attractor_cache,
            force_attractor_refresh=args.force_attractor_refresh,
            node_names=node_names,
            input_values=initial_values,
            use_statistical_attractors=not args.no_statistical_attractors,
            statistical_warmup=args.statistical_warmup,
            statistical_steps=args.statistical_steps,
            statistical_min_attractors=args.statistical_min_attractors,
            statistical_trajectories=args.statistical_trajectories,
            statistical_validation_steps=args.statistical_validation_steps,
            statistical_require_validation=args.statistical_require_validation,
            runtime_attractor_discovery=args.runtime_attractor_discovery,
            runtime_attractor_threshold=args.runtime_attractor_threshold,
            runtime_attractor_trajectories=args.runtime_attractor_trajectories,
            runtime_attractor_max_additions_per_step=args.runtime_attractor_max_additions_per_step,
            runtime_attractor_max_attractors=args.runtime_attractor_max_attractors,
        )
    finally:
        if bang_file != args.assa_file:
            Path(bang_file).unlink(missing_ok=True)

    if args.debug_attractors:
        debug_attractors(env, save_path=args.debug_attractors_save)
        env.save_attractor_cache()
        env.close()
        return

    top_level_log_dir = Path(args.log_dir)
    top_level_log_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.env}_pbn{args.size}_{args.exp_name}"
    checkpoint_path = Path(args.checkpoint_dir) / run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    state_len = env.observation_space.shape[0]
    model = GATTACA(state_len, state_len + 1, config, env)

    model_path = args.model_path
    if args.resume_training and model_path is None:
        latest = get_latest_checkpoint(checkpoint_path)
        model_path = str(latest) if latest is not None else None

    if model_path is not None:
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
    else:
        model.to(device=config.device)

    run = wandb.init(
        project="pbn-rl",
        sync_tensorboard=True,
        monitor_gym=True,
        config={
            "env_backend": "bang",
            "model_name": args.model_name,
            "assa_file": args.assa_file,
            "bang_device": args.bang_device,
            "attractor_device": args.attractor_device,
            "bang_update_type": args.bang_update_type,
            "n_parallel": args.n_parallel,
            "train_parallel": args.train_parallel or args.n_parallel,
            "settle_steps": args.settle_steps,
            "pasip_history_len": args.pasip_history_len,
            "pasip_size": args.pasip_size,
            "runtime_attractor_discovery": args.runtime_attractor_discovery,
            "runtime_attractor_threshold": args.runtime_attractor_threshold,
            "runtime_attractor_trajectories": args.runtime_attractor_trajectories,
            "runtime_attractor_max_additions_per_step": args.runtime_attractor_max_additions_per_step,
            "runtime_attractor_max_attractors": args.runtime_attractor_max_attractors,
            "batch_size": args.batch_size,
            "memory_size": args.memory_size,
            "learning_starts": args.learning_starts,
            "time_steps": args.time_steps,
            "bins": args.bins,
        },
        name=run_name,
        save_code=True,
        mode=args.wandb_mode,
    )

    print(checkpoint_path)
    model.learn(env=env, path=checkpoint_path, wandb=run)

    print(f"final pseudo-attractors were ({len(env.all_attractors)})")
    print(f"final source attractors were ({len(env.divided_attractors)})")
    print(f"final target attractors were ({len(env.target_attractors)})")
    env.save_attractor_cache()
    print("skip testing the model")

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
