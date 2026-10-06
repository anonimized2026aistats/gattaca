import math
import random
from collections import defaultdict

import pandas as pd

from .bittner import base
from gym_PBN.envs.pbn_target_multi import PBNTargetMultiEnv


def graph_from_predictors(df):
    graph = base.Graph(2)  # BN = base 2 values

    nodes = []
    for i, row in df.iterrows():
        promoters = row["Promoters"]
        inhibitors = row["Inhibitors"]

        # check for nans
        promoters = promoters.replace(' ', '').split(',')
        inhibitors = inhibitors.replace(' ', '').split(',')

        promoters = [] if '' in promoters else promoters
        inhibitors = [] if '' in inhibitors else inhibitors

        predictors = promoters + inhibitors
        for p in predictors:
            if ' ' in p:
                print(predictors)
                raise ValueError
        A = [1] * len(promoters) + [-1] * len(inhibitors)
        A.append(1)

        node = base.Node(i, -1, row["Node"], row["Node"])
        node.add_predictors([(1, A, predictors)])
        nodes.append(node)

    graph.add_nodes(nodes)
    return graph


class PBNGTEx(PBNTargetMultiEnv):

    NAME = "GTEx-72"

    def __init__(
        self,
        N=72,
        render_mode: str = "human",
        render_no_cache: bool = False,
        name: str = NAME,
        horizon: int = 100,
        end_episode_on_success: bool = True,
        min_attractors=3,
    ):
        self.N = N
        print(f"its me, gtex-{self.N}")
        if not name:
            name = self.NAME

        df = pd.read_excel(io='~/Downloads/pnas.1722609115.sd02.xlsx')
        df.fillna('', inplace=True)
        graph = graph_from_predictors(df)
        self.path = f"attractors/{self.N}_1_attractors.pkl"

        super().__init__(
            graph,
            {},
            render_mode,
            render_no_cache,
            name,
            None,
            end_episode_on_success,
            min_attractors=min_attractors
        )

        self.all_attractors = [[s] for s in self.statistical_attractors()]

        self.attractor_count = len(self.all_attractors)
        self.probabilities = [1 / self.attractor_count] * self.attractor_count

        print(self.all_attractors)

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attracting_states
