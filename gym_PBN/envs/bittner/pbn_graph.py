import random
from typing import List, Tuple

from .base import Node, Graph
from gym_PBN.utils.converters import logic_funcs_to_PBN_data
from gym_PBN.utils.logic.eval import LogicExpressionEvaluator


class PBNNode(Node):
    def __init__(self, index, data, genes):
        input_mask, truth_table, node, _ = data
        super(PBNNode, self).__init__(index, -1, node, node, False)

        self.input_mask = input_mask
        self.truth_table = truth_table

        predictor_genes = []
        for i in range(len(self.input_mask)):
            if self.input_mask[i]:
                predictor_genes.append(i)

        self.predictors = [predictor_genes]
        self.used_by_agent = False

    def step(self, state, verbose=False):
        relevant_nodes = tuple(state[i] for i in self.predictors[0])
        proba = self.truth_table[relevant_nodes]
        # this requires some finese in case of probabilistic boolean models
        # but works just fine for non-probabilistic ones
        self.value = int(proba)
        return self.value


class PBNGraph(Graph):
    def __init__(self, genes, logic_functions):
        super(PBNGraph, self).__init__(2)
        self.genes = genes
        pbn_data = logic_funcs_to_PBN_data(self.genes, logic_functions)

        for i, data in enumerate(pbn_data):
            print(f'  spawning node {i}')
            self.nodes.append(PBNNode(i, data, self.genes))

    def step(self, changed_nodes: list = None, i=None):
        oldState = self.getState()
        i = random.randint(0, len(self.nodes) - 1) if i is None else i
        # while i in changed_nodes:
        #     i = random.randint(0, len(self.nodes) - 1)
        self.nodes[i].step(oldState)
        return self.getState()

    def get_adj_list(self):
        top_nodes = []
        bot_nodes = []

        for top_node in self.nodes:
            done = set()
            top_nodes.append(top_node.index)
            bot_nodes.append(top_node.index)

            # print(top_node.predictors)

            for bot_node_id in top_node.predictors[0]:
                if bot_node_id not in done:
                    done.add(bot_node_id)
                    top_nodes.append(top_node.index)
                    bot_nodes.append(bot_node_id)

        return [top_nodes, bot_nodes]
