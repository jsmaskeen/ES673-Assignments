import numpy as np
import torch
from scipy.spatial.distance import cdist
import pandas as pd


def get_microns_priors(n_in, n_out, seed=42, lambda_constant=150.0):
    soma_cols = [
        "id",
        "valid",
        "pt_supervoxel_id",
        "pt_root_id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "volume",
    ]
    svm_cols = [
        "id",
        "valid",
        "pt_supervoxel_id",
        "pt_root_id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "classification_system",
        "cell_type",
    ]

    soma_df = pd.read_csv("data/microns/nucleus_detection_v0.csv", names=soma_cols)
    neuron_df = pd.read_csv("data/microns/nucleus_neuron_svm.csv", names=svm_cols)
    cell_type_df = pd.read_csv(
        "data/microns/baylor_log_reg_cell_type_coarse_v1.csv"
    ) 

    merged_df = soma_df.merge(neuron_df[["id", "cell_type"]], on="id")
    neurons_only = merged_df[merged_df["cell_type"] == "neuron"]

    final_df = neurons_only.merge(
        cell_type_df[["target_id", "cell_type"]],
        left_on="id",
        right_on="target_id",
        suffixes=("_svm", "_baylor"),
    )

    total_neurons = n_in + n_out
    if len(final_df) < total_neurons:
        raise ValueError(
            f"Requested {total_neurons} neurons, but only {len(final_df)} available."
        )

    sampled_neurons = final_df.sample(n=total_neurons, random_state=seed)

    resolution = np.array([0.004, 0.004, 0.040])
    x = sampled_neurons["pt_position_x"].values
    y = sampled_neurons["pt_position_y"].values
    z = sampled_neurons["pt_position_z"].values
    positions_um = np.column_stack((x, y, z)) * resolution

    is_excitatory = (sampled_neurons["cell_type_baylor"] == "excitatory").values

    layer_in_pos = positions_um[:n_in]
    layer_out_pos = positions_um[n_in : n_in + n_out]

    pre_is_exc = torch.tensor(is_excitatory[:n_in], dtype=torch.bool)
    post_is_exc = torch.tensor(is_excitatory[n_in : n_in + n_out], dtype=torch.bool)

    distance_matrix_um = cdist(layer_out_pos, layer_in_pos)

    attenuation_matrix = np.exp(-distance_matrix_um / lambda_constant)
    attenuation_matrix[distance_matrix_um > 5 * lambda_constant] = 0.0

    structural_adjacency = torch.tensor(
        (distance_matrix_um < 2 * lambda_constant).astype(np.float32)
    )

    return {
        "attenuation_matrix": torch.tensor(attenuation_matrix, dtype=torch.float32),
        "adjacency_matrix": structural_adjacency,
        "pre_is_exc": pre_is_exc,
        "post_is_exc": post_is_exc,
    }
