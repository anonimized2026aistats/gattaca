import itertools
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
from numba import njit
from tqdm.contrib.concurrent import process_map


def generate_predictor_sets(
        gene_data: pd.DataFrame,
        k=3,
        n_predictors=5,
        savepath="predictor_sets.pkl",
):
    gene_data = gene_data.drop("Name", axis=1)
    n_samples = len(gene_data.columns)

    # Load if exists
    if Path(savepath).exists():
        return pickle.load((open(savepath, "rb")))

    # Otherwise generate
    genes = gene_data.index.unique()
    _func = partial(_gen_predictor_sets_gene, gene_data, n_samples, n_predictors, k)

    predictor_sets = process_map(
        _func,
        genes,
        max_workers=multiprocessing.cpu_count(),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}]",
    )

    # Save
    pickle.dump(predictor_sets, open(savepath, "wb"))

    return predictor_sets


def _gen_predictor_sets_gene(gene_data, n_samples, n_predictors, k, gene):
    # Data processing
    buff = np.empty((3, n_predictors), dtype=object)
    temp_data = gene_data.drop(gene)

    remaining_genes = temp_data.index.unique().to_numpy()
    remaining_genes_idx = range(len(remaining_genes))

    gene_states = np.atleast_2d(gene_data.loc[gene])
    remaining_genes_states = np.array(
        [np.atleast_2d(gene_data.loc[_gene]) for _gene in remaining_genes], dtype=object
    )

    # All possible predictor combinations - O(k * (n choose k))
    for _, combination in enumerate(itertools.combinations(remaining_genes_idx, k)):
        n_genes = len(combination)
        comb_gene_states = remaining_genes_states[list(combination)]

        # sample, gene, state_combo
        x_combos = np.empty((n_samples, n_genes, 0))

        for gene_state_combo in itertools.product(*comb_gene_states):
            states = np.array(gene_state_combo)
            # sample, gene, value
            x = states.T.reshape(-1, n_genes, 1)
            x_combos = np.append(x_combos, x, axis=2)

        for _, _exp in enumerate(gene_states):
            _y = np.expand_dims(_exp, axis=1)
            for state_combo_idx in range(x_combos.shape[2]):
                x = x_combos[:, :, state_combo_idx]
                COD, A = gen_COD(x, _y)
                add_to_buff(buff, (COD, A, remaining_genes[list(combination)]))

    return buff


def add_to_buff(buff, data):
    n_predictors = buff.shape[1]
    COD, _, _ = data

    for i in range(n_predictors):
        # If there's an empty slot in the buffer
        if buff[0, i] is None:
            buff[:, i] = data  # Just add it
            break
        elif buff[0, i] < COD:  # If this is a better predictor
            temp = np.copy(buff[:, i])  # Copy existing data
            buff[:, i] = data  # Overwrite

            # while i < n_predictors - 1:
            # Move everything to the right
            for j in range(i, n_predictors - 1):
                temp2 = np.copy(buff[:, j + 1])
                buff[:, j + 1] = temp
                temp = temp2

            break


def gen_COD(X, Y):
    """Schmulevich's (?) method for generating a COD. The closed form solution."""
    # Need float everywhere for Numba
    ones = np.ones(Y.shape, dtype=np.float_)
    X = np.append(X, ones, axis=1).astype(np.float_)
    Y = Y.astype(np.float_)
    R = np.dot(X.T, X)
    Rp = np.linalg.pinv(R)
    C = np.dot(X.T, Y)
    A = np.dot(Rp, C)  # for comparison

    y_pred = np.dot(X, A).round()
    y_pred_null = ((ones * np.mean(Y)).round()).astype(int) + 10 ** -8
    e_null = mean_squared_error(y_pred_null, Y)

    e = mean_squared_error(y_pred, Y)
    COD = (e_null - e) / e_null
    if COD < 0:
        COD = 10 ** -8
    return COD, A
