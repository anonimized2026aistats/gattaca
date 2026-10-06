from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from gym_PBN.envs.bittner import base, gen
from gym_PBN.envs.bittner.gen.binarise import binarise


def extract_gene_data(file: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Getting gene data
    gene_df = pd.read_excel(
        file, header=[0, 1], sheet_name="CUTANEOUS MELANOMA", skipfooter=5
    )

    gene_ids = gene_df["Clone Data"]["Image Clone ID"]
    gene_ids.name = "ID"

    gene_names = gene_df["Clone Data"]["UniGene Cluster Title"]
    gene_names.name = "Name"

    exp_data = pd.concat(
        (
            gene_df["Ratio Data for Group of 12 Unclustered Cutaneous Melanomas"],
            gene_df["Ratio Data for Cluster of 19 Cutaneous Melanomas"],
        ),
        axis=1,
    )
    exp_data.columns = [f"T{i + 1}" for i in range(len(exp_data.columns))]

    gene_data = pd.concat((gene_ids, gene_names, exp_data), axis=1)
    gene_data = gene_data.set_index("ID")

    # Getting weights
    weight_df = pd.read_excel(file, header=[0, 1], sheet_name="WEIGHTED GENE LIST")
    weight_ids = weight_df["Clone Data"]["Image Clone ID"]
    weight_ids.name = "ID"

    return gene_data, weight_ids


def pad_ids(current_ids: List[str], pad_to: int, id_pool: List[str]) -> List[str]:
    new_ids = [_id for _id in current_ids]

    for _id in id_pool:
        if not _id in new_ids:
            new_ids.append(_id)
            if len(new_ids) == pad_to:
                break

    return new_ids


def spawn(
    file,
    total_genes,
    include_ids,
    bin_method,
    n_predictors=5,
    k=3,
    predictor_sets_path=Path(__file__).parent / "data",
):
    gene_data, weight_ids = extract_gene_data(file)

    if int(total_genes) > len(include_ids):
        include_ids = pad_ids(include_ids, total_genes, weight_ids)
        # we need include_ids to be sorted for interoperability with cabean
        include_ids = sorted(include_ids)

    gene_data = gene_data.loc[include_ids]  # Trim
    gene_data = binarise(gene_data, bin_method)  # Binarise
    gene_data = gene_data.drop_duplicates()  # Drop duplicate rows

    graph = base.Graph(2)  # BN = base 2 values
    predictor_sets = gen.generate_predictor_sets(
        gene_data,
        n_predictors=n_predictors,
        k=k,
        savepath=f"{predictor_sets_path}/predictor_sets_{len(include_ids)}_{n_predictors}_{k}_{bin_method}.pkl",
    )

    nodes = []
    for i, _id in enumerate(gene_data.index.unique()):
        data = gene_data.loc[_id]
        global_index = np.where(gene_data.index == _id)[0][0]

        node = base.Node(i, global_index, data["Name"], _id)
        node.add_predictors(predictor_sets[i].T)
        nodes.append(node)

    graph.add_nodes(nodes)
    return graph
