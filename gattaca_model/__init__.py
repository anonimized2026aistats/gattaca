from collections import defaultdict

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch.distributions import Categorical

from pathlib import Path

import numpy as np
import gym
import random

from .network import GraphBranchingQNetwork

from .memory import ExperienceReplay, PrioritisedER, Transition


class GATTACA(nn.Module):

    def __init__(self, observation, ac, config, env):
        if type(observation) == tuple:
            observation, _ = observation
        super().__init__()

        self.EPSILON = config.epsilon_start
        self.env = env
        self.bins = config.bins
        self.state_size= observation
        print(observation)

        self.action_count = ac

        self.q = GraphBranchingQNetwork(observation, ac, config.bins).to(device=config.device)
        self.target = GraphBranchingQNetwork(observation, ac, config.bins).to(device=config.device)

        self.target.load_state_dict(self.q.state_dict())

        # model_path = 'models/laptop1_pbn28_backprop_reward/bdq_final.pt'
        # self.load_state_dict(torch.load(model_path))

        self.target_net_update_freq = config.target_net_update_freq
        self.config = config
        self.gamma = config.gamma
        self.update_counter = 0

        self.time_steps = 0
        self.start_predicting = config.learning_starts
        self.reward_discount_rate = config.reward_discount_rate

        self.MIN_EPSILON = config.epsilon_final
        self.MAX_EPSILON = config.epsilon_start
        self.EPSILON_DECREMENT = (self.MAX_EPSILON - self.MIN_EPSILON) / config.epsilon_decay
        self.missed_paths = 0

        self.edge_index = self.get_adj_list()

        self.wandb = None

        self.attractor_count = len(env.attracting_states)

    def dst(self, l1, l2):
        ret = 0
        for x, y in zip(l1, l2):
            if x != y:
                ret += 1
        return ret

    def predict(self, state, state2):
        with torch.no_grad():
            # exploration probability
            epsilon = self.decrement_epsilon()
            state_array = np.asarray(state)
            is_batched = state_array.ndim == 2

            if np.random.random() < epsilon:
                state_count = state_array.shape[0] if is_batched else 1
                white_list = list(set(range(self.state_size)) - set(self.env.forbidden_actions))

                # while len(diff_list) == 0:
                #     target = random.choice(self.env.target_attractors)
                #     # action_len = random.randint(1, self.config.bins)
                #     action_len = self.config.bins
                #
                #     diff_list = [x for x in white_list if state[x] != target[x]]
                #
                # if len(diff_list) == 0:
                #     print(self.env.in_target(state))
                #     for a in self.env.target_attractors:
                #         print(a)
                #
                #     print('------------------------')
                #     print(state)
                #
                # if len(diff_list) < action_len:
                #     action = [1 + x for x in diff_list]
                #     action += [0] * (action_len - len(diff_list))
                # else:
                actions = [
                    [1 + x for x in random.sample(white_list, self.config.bins)]
                    for _ in range(state_count)
                ]
                if is_batched:
                    action = torch.tensor(actions, device=self.config.device)
                else:
                    action = torch.tensor(actions[0], device=self.config.device)
            else:
                # s = np.stack((state, target))
                if is_batched:
                    x = torch.tensor(state_array, dtype=torch.float, device=self.config.device)
                    x = torch.stack((x, x), dim=2)
                else:
                    x = torch.tensor((state, state), dtype=torch.float, device=self.config.device)
                    x = x.t()
                    x = x.unsqueeze(dim=0)

                out = self.q(x, self.edge_index)

                # for i in self.env.forbidden_actions:
                #     out[:, i+1] = 0

                action = torch.argmax(out, dim=2).to(self.config.device)
                if not is_batched:
                    action = action.squeeze(0)

            return action

    def learn_vectorized(self, env, path, wandb):
        config = self.config
        memory = ExperienceReplay(config.memory_size)
        adam = optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.wandb = wandb

        state, _ = env.reset()
        episode_rewards = np.zeros(env.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(env.num_envs, dtype=np.int32)
        rew_recap = []
        len_recap = []

        p_bar = tqdm(total=config.time_steps)
        missed = defaultdict(int)
        completed_successes = 0
        completed_truncations = 0

        for frame in range(config.time_steps):
            action = self.predict(state, state)
            new_state, reward, terminated, truncated, infos = env.step(action)
            done = np.logical_or(terminated, truncated)

            action_rows = action.detach().cpu()
            for env_index in range(env.num_envs):
                memory.store(Transition(
                    state[env_index],
                    state[env_index],
                    action_rows[env_index].to(self.config.device),
                    float(reward[env_index]),
                    new_state[env_index],
                    bool(done[env_index])
                ))

            episode_rewards += reward
            episode_lengths += 1

            if np.any(truncated):
                truncated_count = int(np.sum(truncated))
                missed[("vectorized", "truncated")] += truncated_count
                completed_truncations += truncated_count
            completed_successes += int(np.sum(terminated))

            if len(self.env.all_attractors) > self.attractor_count:
                self.attractor_count = len(self.env.all_attractors)

            if np.any(done):
                done_indices = np.flatnonzero(done)
                rew_recap.extend(episode_rewards[done_indices].tolist())
                len_recap.extend(episode_lengths[done_indices].tolist())

                wandb.log({
                    "episode_len": float(np.mean(episode_lengths[done_indices])),
                    "episode_reward": float(np.mean(episode_rewards[done_indices])),
                    "completed_episodes": int(len(done_indices)),
                })

                reset_state, _ = env.reset(indices=done_indices)
                new_state[done_indices] = reset_state[done_indices]
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

            state = new_state
            p_bar.update(1)

            if frame > max(config.batch_size, config.learning_starts):
                self.update_policy(adam, memory, None, config.batch_size)

            if frame % 1000 == 0:
                self.missed_paths = sum(missed.values())
                avg_reward = float(np.average(rew_recap)) if len(rew_recap) > 0 else float("nan")
                avg_len = float(np.average(len_recap)) if len(len_recap) > 0 else float("nan")
                print(missed)
                print(f"Average episode reward: {avg_reward}")
                print(f"Avg len: {avg_len}")
                completed_total = completed_successes + completed_truncations
                success_rate = (
                    completed_successes / completed_total
                    if completed_total > 0
                    else float("nan")
                )
                print(f"Success rate: {success_rate:.3f}")

                wandb.log({"Avg episode reward": avg_reward,
                           "Avg episode length": avg_len,
                           "Success rate": success_rate,
                           "Attracting state count": self.attractor_count,
                           "Exploration probability": self.EPSILON,
                           "Missed paths": sum(missed.values()),
                           "Parallel trajectories": env.num_envs})

                missed.clear()
                completed_successes = 0
                completed_truncations = 0
                rew_recap = []
                len_recap = []
                self.save(f"{path}/bdq_{frame}.pt")
        self.save(f"{path}/bdq_final.pt")

    def update_policy(self, adam, memory, memory_negative, batch_size):
        x = memory.sample(batch_size)
        # x_neg = memory_negative.sample(min(batch_size, len(memory_negative)))
        # x = x_pos + x_neg
        b_states, b_targets, b_actions, b_rewards, b_next_states, b_masks = zip(*x)

        states = torch.tensor(np.stack(b_states), device=self.config.device).float()
        # targets = torch.tensor(np.stack(b_targets), device=self.config.device).float()
        actions = torch.stack(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(np.stack(b_rewards), device=self.config.device).float().reshape(-1, 1)
        next_states = torch.tensor(np.stack(b_next_states), device=self.config.device).float()
        dones = torch.tensor(np.stack(b_masks), device=self.config.device).float().reshape(-1, 1)

        input_tuples = torch.stack((states, states), dim=2)
        qvals = self.q(input_tuples, self.edge_index)

        current_q_values = qvals.gather(2, actions).squeeze(-1)

        with torch.no_grad():
            next_input_tuple = torch.stack((next_states, next_states), dim=2)
            argmax = torch.argmax(self.q(next_input_tuple, self.edge_index), dim=2)

            max_next_q_vals = self.target(next_input_tuple, self.edge_index).gather(2, argmax.unsqueeze(2)).squeeze(-1)

        expected_q_vals = rewards + max_next_q_vals * self.gamma * (1.0 - dones)
        loss = F.mse_loss(expected_q_vals, current_q_values)
        self.wandb.log({"loss": loss.data})

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-10., 10.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())

    def decrement_epsilon(self):
        """Decrement the exploration rate."""
        self.time_steps += 1

        if self.time_steps > self.start_predicting:
            if self.time_steps % 20_000 == 0:
                if self.missed_paths > 35:
                    self.EPSILON = max(self.EPSILON, 0.1 + self.EPSILON_DECREMENT)

            self.EPSILON = max(self.MIN_EPSILON, self.EPSILON - self.EPSILON_DECREMENT)

        return self.EPSILON

    def learn(self,
              env,
              path,
              wandb,
              ):

        config = self.config
        self.pos_bs = config.batch_size
        self.neg_bs = config.batch_size

        memory = ExperienceReplay(config.memory_size)
        # memory_negative = ExperienceReplay(config.memory_size)
        adam = optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.wandb = wandb

        if getattr(env, "is_vectorized", False):
            return self.learn_vectorized(env, path, wandb)

        state, _ = env.reset()
        ep_reward = 0.
        ep_len = 0
        recap = []
        rew_recap = []
        len_recap = []

        p_bar = tqdm(total=config.time_steps)
        missed = defaultdict(int)
        transitions = []

        for frame in range(config.time_steps):

            action = self.predict(state, state)

            env_action = list(action.unique())
            new_state, reward, terminated, truncated, infos = env.step(env_action)
            done = terminated | truncated

            if terminated:
                for _ in range(1):
                    memory.store(Transition(
                        state,
                        state,
                        action,
                        reward,
                        new_state,
                        done
                    ))
            else:
                memory.store(Transition(
                    state,
                    state,
                    action,
                    reward,
                    new_state,
                    done
                ))

            if truncated:
                missed[(self.env.state_attractor_id, self.env.target_attractor_id)] += 1

            if len(self.env.all_attractors) > self.attractor_count:
                self.attractor_count = len(self.env.all_attractors)
                # self.EPSILON = max(self.EPSILON, 0.3)

            ep_len += 1

            if done:
                # we need to propagate reward along whole path
                ep_reward = reward

                new_state, _ = env.reset()

                recap.append(ep_reward)
                p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
                rew_recap.append(ep_reward)
                len_recap.append(ep_len)
                wandb.log({"episode_len": ep_len,
                           "episode_reward": ep_reward})
                ep_reward = 0.
                ep_len = 0

            state = new_state

            p_bar.update(1)

            if frame > max(config.batch_size, config.learning_starts):
                self.update_policy(adam, memory, None, config.batch_size)

            if frame % 1000 == 0:
                self.missed_paths = sum(missed.values())
                print(missed)
                print(f"Average episode reward: {np.average(rew_recap)}")
                print(f"Avg len: {np.average(len_recap)}")

                wandb.log({"Avg episode reward": np.average(rew_recap),
                           "Avg episode length": np.average(len_recap),
                           "Attracting state count": self.attractor_count,
                           "Exploration probability": self.EPSILON,
                           "Missed paths": sum(missed.values())})

                # env.env.evn.env.rework_probas_epoch(len_recap)
                missed.clear()
                rew_recap = []
                len_recap = []
                self.save(f"{path}/bdq_{frame}.pt")
        self.save(f"{path}/bdq_final.pt")

    def save(self, path):
        print(path)
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def get_adj_list(self):
        # env = self.env
        # top_nodes = []
        # bot_nodes = []
        #
        # for top_node in env.graph.nodes:
        #     done = set()
        #     top_nodes.append(top_node.index)
        #     bot_nodes.append(top_node.index)
        #
        #     print(top_node.index, top_node.predictors)
        #     for predictor, _, _ in top_node.predictors:
        #         for bot_node_id in predictor:
        #             if bot_node_id not in done:
        #                 done.add(bot_node_id)
        #                 top_nodes.append(top_node.index)
        #                 bot_nodes.append(env.graph.getNodeByID(bot_node_id).index)
        #
        # return torch.tensor([top_nodes, bot_nodes], dtype=torch.long, device=self.config.device)
        return torch.tensor(self.env.graph.get_adj_list(), dtype=torch.long, device=self.config.device)
