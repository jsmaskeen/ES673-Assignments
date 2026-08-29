import torch
import torch.nn as nn
from .synapse import SparseDendriticSynapse
from .neuron import BioPerceptronNeuron


class ModifiedPerceptronClassifier(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=128,
        output_dim=10,
        priors_l1=None,
        priors_l2=None,
        weight_scale=10.0,
        tau_m=20.0,
        U=0.2,
        gamma=0.02,
        surrogate_k=25.0,
    ):
        super().__init__()

        # Layer 1 (is_sensory_input=True)
        self.synapse1 = SparseDendriticSynapse(
            input_dim,
            hidden_dim,
            priors_l1,
            is_sensory_input=True,
            weight_scale=weight_scale,
            U=U,
        )
        self.layer1 = BioPerceptronNeuron(
            input_dim, hidden_dim, priors_l1, tau_m=tau_m, gamma=gamma, surrogate_k=surrogate_k
        )
        self.layer1.synapse = self.synapse1

        # Layer 2 is an internal layer
        self.synapse2 = SparseDendriticSynapse(
            hidden_dim,
            output_dim,
            priors_l2,
            is_sensory_input=False,
            weight_scale=weight_scale,
            U=U,
        )
        self.layer2 = BioPerceptronNeuron(
            hidden_dim, output_dim, priors_l2, tau_m=tau_m, gamma=gamma, surrogate_k=surrogate_k
        )
        self.layer2.synapse = self.synapse2
        self.layer2.synapse = self.synapse2

    def forward(self, input_spikes, return_states=False):
        # input_spikes: (Batch, Time, Features)

        # Layer 1
        hidden_spikes = self.layer1(input_spikes)

        # Layer 2
        if return_states:
            out_spikes, l2_states = self.layer2(hidden_spikes, return_states=True)
            return torch.sum(out_spikes, dim=1), out_spikes, hidden_spikes, l2_states
        else:
            out_spikes = self.layer2(hidden_spikes)
            # Prediction is the accumulated spike count
            return torch.sum(out_spikes, dim=1), out_spikes
