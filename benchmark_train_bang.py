import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from gattaca_model import GATTACA
from gattaca_model.memory import ExperienceReplay, Transition
from train_bang import (
    TrainingPBNBangEnv,
    build_config,
    patch_bang_merge_attractors,
    prepare_bang_source,
)


class NullRun:
    def log(self, *_args, **_kwargs):
        pass


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def parse_sizes(value):
    return [int(item) for item in value.split(",")]


def build_env(args, parallel):
    bang_file, bang_format, input_nodes, initial_values, target_nodes, target_values, node_names = (
        prepare_bang_source(args.assa_file, args.bang_format, args.model_name)
    )
    try:
        return TrainingPBNBangEnv(
            file=bang_file,
            format=bang_format,
            model_name=None,
            input_nodes=input_nodes,
            target_nodes=target_nodes,
            target_values=target_values,
            name=args.model_name,
            horizon=100,
            n_parallel=parallel,
            pasip_history_len=args.pasip_history_len,
            pasip_size=args.pasip_size,
            settle_steps=args.settle_steps,
            bang_device=args.bang_device,
            attractor_device=args.attractor_device,
            train_parallel=parallel,
            attractor_cache=args.attractor_cache,
            node_names=node_names,
            input_values=initial_values,
            use_statistical_attractors=False,
        )
    finally:
        if bang_file != args.assa_file:
            Path(bang_file).unlink(missing_ok=True)


def benchmark(args, parallel, batch_size):
    if args.bang_device == "cuda" and parallel % 32 != 0:
        raise ValueError("CUDA parallel sizes must be multiples of 32")

    env = build_env(args, parallel)
    args.batch_size = batch_size
    args.learning_starts = batch_size
    args.memory_size = max(args.memory_size, batch_size * 2)
    args.time_steps = args.iterations
    args.bins = 5
    args.no_cuda = False

    config = build_config(args)
    model = GATTACA(env.observation_space.shape[0], env.observation_space.shape[0] + 1, config, env)
    model.to(config.device)
    model.wandb = NullRun()
    model.EPSILON = 0.0
    model.MIN_EPSILON = 0.0

    optimizer = optim.Adam(model.q.parameters(), lr=config.learning_rate)
    memory = ExperienceReplay(config.memory_size)
    state, _ = env.reset()

    totals = {"predict": 0.0, "step": 0.0, "replay": 0.0, "update": 0.0}
    update_count = 0

    for iteration in range(args.warmup + args.iterations):
        action, predict_time = timed(config.device, lambda: model.predict(state, state))
        step_result, step_time = timed(config.device, lambda: env.step(action))
        new_state, reward, terminated, truncated, _ = step_result
        done = np.logical_or(terminated, truncated)

        replay_start = time.perf_counter()
        action_rows = action.detach().cpu()
        for env_index in range(env.num_envs):
            memory.store(
                Transition(
                    state[env_index],
                    state[env_index],
                    action_rows[env_index].to(config.device),
                    float(reward[env_index]),
                    new_state[env_index],
                    bool(done[env_index]),
                )
            )
        replay_time = time.perf_counter() - replay_start

        update_time = 0.0
        measured_iteration = iteration >= args.warmup
        if len(memory) >= batch_size:
            _, update_time = timed(
                config.device,
                lambda: model.update_policy(optimizer, memory, None, batch_size),
            )
            if measured_iteration:
                update_count += 1

        if np.any(done):
            reset_state, _ = env.reset(indices=np.flatnonzero(done))
            new_state[done] = reset_state[done]
        state = new_state

        if measured_iteration:
            totals["predict"] += predict_time
            totals["step"] += step_time
            totals["replay"] += replay_time
            totals["update"] += update_time

    total_time = sum(totals.values())
    transitions = args.iterations * parallel
    result = {
        "parallel": parallel,
        "batch_size": batch_size,
        "iterations": args.iterations,
        "updates": update_count,
        "seconds": total_time,
        "iterations_per_second": args.iterations / total_time,
        "transitions_per_second": transitions / total_time,
        "update_samples_per_second": (
            update_count * batch_size / totals["update"] if totals["update"] else 0.0
        ),
        **{f"{name}_seconds": value for name, value in totals.items()},
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark vectorized BANG training throughput.")
    parser.add_argument("--assa-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--bang-format", choices=["ispl", "bnet", "assa", "sbml"], default="ispl")
    parser.add_argument("--attractor-cache", required=True)
    parser.add_argument("--parallel", default="256,512,1024,2048")
    parser.add_argument("--batch-sizes", default="256,512,1024")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--settle-steps", type=int, default=1000)
    parser.add_argument("--pasip-history-len", type=int, default=5)
    parser.add_argument("--pasip-size", type=int, default=2)
    parser.add_argument("--bang-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--attractor-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="training_throughput.csv")
    parser.add_argument("--worker-parallel", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    patch_bang_merge_attractors()

    if args.worker_parallel is not None:
        result = benchmark(args, args.worker_parallel, args.worker_batch_size)
        with open(args.worker_output, "w") as output_file:
            json.dump(result, output_file)
        return

    results = []
    for parallel in parse_sizes(args.parallel):
        for batch_size in parse_sizes(args.batch_sizes):
            print(f"\nBenchmarking parallel={parallel}, batch_size={batch_size}", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as result_file:
                result_path = Path(result_file.name)

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--assa-file", args.assa_file,
                "--model-name", args.model_name,
                "--bang-format", args.bang_format,
                "--attractor-cache", args.attractor_cache,
                "--iterations", str(args.iterations),
                "--warmup", str(args.warmup),
                "--memory-size", str(args.memory_size),
                "--settle-steps", str(args.settle_steps),
                "--pasip-history-len", str(args.pasip_history_len),
                "--pasip-size", str(args.pasip_size),
                "--bang-device", args.bang_device,
                "--attractor-device", args.attractor_device,
                "--worker-parallel", str(parallel),
                "--worker-batch-size", str(batch_size),
                "--worker-output", str(result_path),
            ]

            completed = subprocess.run(command)
            if completed.returncode != 0:
                print(
                    f"FAILED parallel={parallel}, batch_size={batch_size}, "
                    f"exit_code={completed.returncode}",
                    flush=True,
                )
                result_path.unlink(missing_ok=True)
                continue

            with open(result_path) as result_file:
                result = json.load(result_file)
            result_path.unlink(missing_ok=True)
            results.append(result)
            print(
                f"{result['iterations_per_second']:.2f} it/s, "
                f"{result['transitions_per_second']:.0f} transitions/s, "
                f"{result['update_samples_per_second']:.0f} update samples/s",
                flush=True,
            )

            with open(args.output, "w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    if not results:
        raise RuntimeError("All benchmark configurations failed")

    with open(args.output, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
