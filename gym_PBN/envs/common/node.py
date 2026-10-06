from gym_PBN.types import INPUT_MASK, TRUTH_TABLE
from numpy import random


class Node:
    def __init__(
        self,
        input_mask: INPUT_MASK,
        function: TRUTH_TABLE,
        i: int,
        name: str = None,
        is_control: bool = False,
    ):
        """represents node in a PBN.

        args:
            mask [Node]: List of node objects that are inputs of this node.
            function [float]: matrix representation of function
            name (String): Name of the gene
            is_control (bool): Whether the node is a control node
        """
        self.input_mask = input_mask
        self.function = function
        self.i = i
        self.name = name if name is not None else f"G{i}"
        self.is_control = is_control

    def value(self, state):
        return state[self.i]

    def get_next_value_prob(self, state):
        return self.function.item(tuple(state[self.input_mask].astype(int)))

    def compute_next_value(self, state):
        """Return own next-state given the particular state according to own function and states of input genes."""
        # Calculate the next value stochastically using the probability of the node going to true given the input state.
        u = random.uniform(0, 1)
        return u < self.get_next_value_prob(state)

    def __str__(self):
        return f"{self.name}{' (Control)' if self.is_control else ''}"
