import copy
import random
import time
from typing import List, Union

import networkx as nx
import numpy as np
from gym_PBN.types import LOGIC_FUNC_DATA, PBN_DATA, STATE, TRUTH_TABLE
from gym_PBN.utils import booleanize
from gym_PBN.utils.converters import logic_funcs_to_PBN_data

from .node import Node


class PBN:
    def __init__(
        self, PBN_data: PBN_DATA = [], logic_func_data: LOGIC_FUNC_DATA = None, goal_config=None
    ):
        """Construct a PBN from given PBN data.

        Args:
            PBN_data (list): data representing the PBN.

        returns:
            PBN
        """
        if len(PBN_data) != 0:
            self._init_from_pbn_data(PBN_data)
        else:
            self._init_from_logic_funcs(logic_func_data)

        print(f"got PBN.N = {self.N}")
        print(self.print_PBN())
        if goal_config is not None:
            self.target_nodes = goal_config["target_nodes"]

    def _init_from_pbn_data(self, PBN_data: PBN_DATA):
        self.N = len(PBN_data)
        self.nodes = np.empty((self.N), dtype=object)
        self.state = np.empty((self.N), dtype=bool)
        self.PBN_graph = None
        self.STG = None

        for i in range(self.N):
            node_data = PBN_data[i]
            self.nodes[i] = Node(*node_data)

    def _logic_funcs_to_pbn_data(self, logic_func_data: LOGIC_FUNC_DATA):
        return logic_funcs_to_PBN_data(*logic_func_data)

    def _init_from_logic_funcs(self, logic_func_data: LOGIC_FUNC_DATA):
        PBN_data = self._logic_funcs_to_pbn_data(logic_func_data)
        self._init_from_pbn_data(PBN_data)

    def reset(
        self, state: Union[List[Union[int, bool]], np.ndarray, None] = None
    ) -> STATE:
        """Set the state of the PBN to a particular one.

        args:
            state [bool]: The state to be set to. If left empty, defaults to a random state.
        """
        self.state = np.array(state)
        if state is None:
            self.state = np.random.rand(self.N) > 0.5
        else:
            if len(state) != self.N:
                raise Exception(
                    f"The length of the state given ({len(state)}) is different from the PBN size ({self.N})."
                )

            self.state = (
                np.array(state, dtype=bool)
                if type(state) != np.ndarray
                else state.astype(bool)
            )
        self.state[0] = 0
        return self.state

    def flip(self, index: int):
        """Flip the value of a gene at index.

        args:
            index (int): gene index to flip.
        """
        self.state[index] = not self.state[index]

    def step(self):
        """Perform a step of natural evolution."""
        i = random.randint(1, len(self.state) - 1)
        node = self.nodes[i]
        self.state[i] = node.compute_next_value(self.state)

        # self.state = np.array(
        #     [node.compute_next_value(self.state) for node in self.nodes], dtype=bool
        # )

    def name_nodes(self, names: List[str]):
        for i in range(self.N):
            self.nodes[i].name = names[i]

    def get_node_by_name(self, name: str) -> Node:
        """Get the appropriate node object given the name of the node."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise Exception(f'Node with name "{name}" not found.')

    def print_functions(self) -> List[TRUTH_TABLE]:
        """Print the functions of the PBN to inspect visually."""
        return [node.function for node in self.nodes]

    def print_PBN(self, no_cache: bool = False) -> nx.DiGraph:
        """Construct a networkx graph representing the connetcivities of the PBN.

        returns: networkx di-graph.
        """
        if self.PBN_graph is None or no_cache:
            G = nx.DiGraph()
            G.add_nodes_from([node.name for node in self.nodes])

            for node in self.nodes:
                input_nodes = self.nodes[node.input_mask]
                G.add_edges_from(
                    [(input_node.name, node.name) for input_node in input_nodes]
                )

            self.PBN_graph = G

        return self.PBN_graph

    def print_STG(self, no_cache: bool = False) -> nx.DiGraph:
        """Generate the State Transition Graph (STG) of the PBN.

        Go through each possible state.
        Compute the probabilities of going to next states.

        returns:
            networkx DiGraph.
        """
        if self.STG is None or no_cache:
            N_states = 2 ** (self.N)

            G = nx.DiGraph()

            # start = time.time()
            for state_index in range(N_states):
                state = booleanize(state_index, self.N)
                G.add_node(str(state.astype(int)))

                next_states = self._compute_next_states(state)
                G.add_weighted_edges_from(next_states)

                # est = N_states * (time.time() - start) / (state_index + 1)
                # print(f"Computing STG: At index {state_index} {state_index * 100 / N_states}%. Est duration: {est}s, OR {est / 60} mins, OR {est / 3600} hrs", end="\r")

            # print(end="\n")
            self.STG = G

        return self.STG

    def _compute_next_states(self, state):
        """Compute the probabilities of going to all next possible states from current state.

        Go through each gene. Compute the probability of each gene being True after current state.
        Convert those probabilities to next possible states.

        args:
            state [bool]: State to calculate probabilities from.

        returns:
            list of triplets. (Current State, next-possible-state, probability.)

        """
        probabilities = np.zeros((2, self.N), dtype=float)

        output = []

        #it's ugly, but i will use it just for testing
        async_mode = True

        # for each node
        # try to update
        # if chage probability is 0 -> skip
        # if it's non zero update
        if async_mode:
            #print("im in async mode")
            for i in range(self.N):
                new_state = state.copy()
                prob_true = self.nodes[i].get_next_value_prob(state)
                if prob_true > 0. and state[i] == 0:
                    new_state[i] = 1
                    output.append((str(state.astype(int)), str(new_state.astype(int)), prob_true))

                if prob_true < 1. and state[i] == 1:
                    new_state[i] = 0
                    output.append((str(state.astype(int)), str(new_state.astype(int)), prob_true))
            #print(output)
            return output
        else:
            #print("im in sync mode")
            # for each node
            for i in range(self.N):
                # probability of activation of node
                prob_true = self.nodes[i].get_next_value_prob(state)
                probs = np.array([1 - prob_true, prob_true])
                probabilities[:, i] = probs

            # probabilities = [prob_neg, prob_true for each node]
            prob_to_states = self._probs_to_states(probabilities)

            for prostate, proprob in prob_to_states:
                # (old state, new state, transisition proba)
                output.append((str(state.astype(int)), str(prostate.astype(int)), proprob))
            #print(output)
            return output

    # probs się źle generują
    def _async_probs_to_states(self, probs):
        """
        args:
            probs: array of pairs prob of state, state for each gene
        """

        _, n_genes = probs.shape

        # n_genes x 1
        # state of PBN
        protostate = np.ones(n_genes, dtype=float) * 0.5
        protostate[0] = 0
        protoprob = 1

        state = self.state

        prostate = [(protostate, protoprob)]
        #print(f"\n\ncalculating for {probs}")

        #select gene to update
        for gene_i in range(1, n_genes):
            # probability of state, state
            p = probs[:, gene_i]
            #print(f"for {gene_i} i got {p}")

            prostate_copy = []

            for i, pro in enumerate(prostate):
                #print(i, pro)
                pro_1, prob_1 = copy.deepcopy(pro)
                pro_2, prob_2 = copy.deepcopy(pro)

                pro_1[gene_i] = 0  # Set value to 0
                pro_2[gene_i] = 1  # Set value to 1
                prob_1 *= p[0]  # Set probability to that value being 0
                prob_2 *= p[1]  # ^^^

                # Put them back in.
                protostate_1 = (pro_1, prob_1)
                protostate_2 = (pro_2, prob_2)
                #print(f"appending {prob_1}:{protostate_1} and {prob_2}:{protostate_2}")
                if prob_1 > 0:
                    prostate_copy.append(protostate_1)
                if prob_2 > 0:
                    prostate_copy.append(protostate_2)
            prostate = prostate_copy
        return prostate


    def _probs_to_states(self, probs):
        """Compute the next possible states to go to, and their probabilities, given a set of probabilities of being true for each gene.

        Set the next states as a list of 0.5 with probability 1.
        A gene can not be at value 0.5, so it is used to signify an uncomputed value.

        Go through each gene.
        If probability is 1 or 0
            set the value of all next states at that index to the particular value. Leave probability unaffected.
        Else,
            Make two copies of all next states - one for each state of the gene. Compute probabilities accordingly.

        args:
            probs [float]: Probabilities of each gene being true in the next state.

        returns:
           List of tuples. Each tuple is a possible next state with according probability.
        """
        _, n_genes = probs.shape

        protostate = np.ones(n_genes, dtype=float) * 0.5
        protoprob = 1

        #print(f"\n\ncalculating for {probs}")

        prostate = [(protostate, protoprob)]
        for gene_i in range(n_genes):
            p = probs[:, gene_i]
            if p[0] == 1 or p[0] == 0:
                # Deterministic. Mainly for optimisation.
                for pro in prostate:
                    protostate, protoprob = pro  # Go through each next-state already computed, unpack them
                    protostate[gene_i] = p[1]  # Set the value of the gene to the corresponding value.
            else:
                prostate_copy = []
                for pro in prostate:
                    pro_1, prob_1 = copy.deepcopy(pro)
                    pro_2, prob_2 = copy.deepcopy(pro)

                    pro_1[gene_i] = 0  # Set value to 0
                    pro_2[gene_i] = 1  # Set value to 1
                    prob_1 *= p[0]  # Set probability to that value being 0
                    prob_2 *= p[1]  # ^^^
                    # Put them back in.
                    protostate_1 = (pro_1, prob_1)
                    protostate_2 = (pro_2, prob_2)
                    prostate_copy.append(protostate_1)
                    prostate_copy.append(protostate_2)
                prostate = prostate_copy
        return prostate
