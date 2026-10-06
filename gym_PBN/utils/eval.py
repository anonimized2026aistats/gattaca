import copy
import itertools
import multiprocessing
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from gym_PBN.envs.dep_pbn_env import PBNEnv
from gym_PBN.envs.pbn_target import PBNTargetEnv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def _bit_seq_to_str(seq: Iterable[int]) -> str:
    return "".join([str(i) for i in seq])


def compute_ssd_hist(
    env: PBNTargetEnv,
    model: object = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
    multiprocess: bool = True,
) -> pd.DataFrame:
    SSD_N = iters  # Number of experiences to sample for the SSD calculation
    SSD_RESETS = resets
    BIT_FLIP_PROB = bit_flip_prob

    assert (
        bit_flip_prob >= 0 and bit_flip_prob <= 1
    ), "Invalid Bit Flip Probability value."
    assert SSD_RESETS > 0, "Invalid resets value."
    assert SSD_N > 0, "Invalid iterations value."
    assert SSD_N // SSD_RESETS, "Resets does not divide the iterations."

    g = len(env.target_nodes)

    if multiprocess:
        _func = partial(_ssd_run, g, SSD_N // SSD_RESETS, BIT_FLIP_PROB, model)
        _iter = [copy.deepcopy(env) for _ in range(SSD_RESETS)]

        max_workers = multiprocessing.cpu_count()
        print(f"Will compute the SSD with {max_workers} processes.")

        all_ssds = process_map(
            _func,
            _iter,
            max_workers=max_workers,
            desc=f"SSD run for {env.name}",
        )

    else:
        all_ssds = []
        for _ in tqdm(range(SSD_RESETS), desc=f"SSD run for {env.name}"):
            all_ssds.append(_ssd_run(g, SSD_N // SSD_RESETS, BIT_FLIP_PROB, model, env))

    print(len(all_ssds))
    print(all_ssds[0])
    ssd = np.array(all_ssds)

    ssd = np.mean(ssd, axis=0)
    ssd /= SSD_N // SSD_RESETS  # Normalize
    ret = ssd

    states = list(map(_bit_seq_to_str, itertools.product([0, 1], repeat=g)))
    ret = pd.DataFrame(list(ssd), index=states, columns=["Value"])
    plot = visualize_ssd(ret, env.name)

    return ret, plot


# This function serves no real purpouse, but removing it is non-trivial
def _ssd_run(g, iters, bit_flip_prob, model, env):
    sub_ssd = np.zeros(2**g, dtype=np.float32)
    env.reset()

    for _ in range(iters):
        state = env.render()
        target = state
        # Convert relevant part of state to binary string, then parse it as an int to get the bucket index.
        bucket = env.render()
        # AM: env.render in the line above does not accept the 'mode' argument. Therefore, a new
        #     getTargetIdx() method
        #
        bucket = env.getTargetIdx()
        sub_ssd[bucket] += 1

        if not model:  # Control the environment
            flip = np.random.rand(len(state)) < bit_flip_prob
            for j in range(len(state)):
                if flip[j]:
                    env.graph.flipNode(j)
            env.step(action=0)
        else:
            action = model.predict(state, target, deterministic=True)
            if type(action) == tuple:
                action = action[0]
            env.step(action=action)

    return sub_ssd


def eval_increase(
    env: PBNTargetEnv,
    model: object,
    original_ssd: pd.DataFrame = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
) -> float:
    """Compute the total increase in the favourable states in the SSD histogram.

    Args:
        env (PBNTargetEnv): the gym-PBN environment.
        model (object): a Stable Baselines model or an Agent of a similar interface.
        original_ssd (pd.DataFrame, optional): the cached uncontrolled SSD Histogram. Defaults to None,
            and if not provided, it will be recalculated.
        iters (int, optional): how many environment transitions to compute. Defaults to 1.2 Million.
        resets (int, optional): how many times to reset the environment. `iters` should be divisible by this number. Defaults to 300.
        bit_flip_prob (float, optional): number in [0,1] on the probability of flipping each bit at random when no control is being applied. Defaults to 0.01

    Returns:
        float: the total increase across all favourable states.
    """
    if original_ssd is None:  # Cache
        original_ssd = compute_ssd_hist(
            env, iters=iters, resets=resets, bit_flip_prob=bit_flip_prob
        )
    model_ssd = compute_ssd_hist(
        env, model, iters=iters, resets=resets, bit_flip_prob=bit_flip_prob
    )
    states_of_interest = [_bit_seq_to_str(state) for state in env.target_node_values]
    return (model_ssd - original_ssd)[states_of_interest].sum()


def visualize_ssd(ssd_frame: pd.DataFrame, env_name: str) -> object:
    """Visualize and save the Steady State Distribution histogram.

    Args:
        ssd_frame (pd.DataFrame): a DataFrame containing the states of interest and
            their corresponding probability.
        env_name (str): the name of the environment for metadata's sake.
    """
    fig = px.bar(
        ssd_frame,
        x=ssd_frame.index,
        y="Value",
        labels={
            "states": "Gene Premutations",
            "ssd_values": "Steady State Distribution",
        },
        title=f"SSD for {env_name}",
    )
    return fig


def eval_winrate(
    env: PBNEnv, model: object, max_states: int = 200_000
) -> tuple[float, float, float]:
    states = itertools.product([0, 1], repeat=env.observation_space.n)

    iters = 0
    wins = 0
    n_interactions = []
    n_timesteps = []
    for i, state in enumerate(states):
        if state in env.target:
            continue
        iters += 1
        observation, _ = env.reset(options={"state": state})
        j = 0
        total_steps = 0
        while True:
            action = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            total_steps += info["interval"]
            j += 1

            if terminated:
                wins += 1
                raise ValueError

            if terminated or truncated:
                n_interactions.append(j)
                n_timesteps.append(total_steps)
                break

        if i > max_states:
            break

    winrate = wins / iters
    avg_interactions = np.mean(n_interactions)
    avg_timesteps = np.mean(n_timesteps)

    return winrate, avg_interactions, avg_timesteps