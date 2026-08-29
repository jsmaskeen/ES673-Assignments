import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseDendriticSynapse(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        priors_dict,
        is_sensory_input=False,
        weight_scale=10.0,
        U=0.2,
    ):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.raw_weights, a=math.sqrt(5))
        self.raw_weights.data *= weight_scale
        self.U = U

        mask = torch.ones(in_features)
        if not is_sensory_input:
            pre_is_exc = priors_dict["pre_is_exc"]
            mask[~pre_is_exc] = -1.0  # Set inhibitory neurons to -1

        self.register_buffer("dale_mask", mask.unsqueeze(0))

        # Homeostatic Normalization
        # This makes the layer invariant to the absolute scale of the attenuation matrix.
        # If lambda is dropped to 50, the raw attenuation values become small,
        # which starves the neurons of current and makes the gradients really small.
        # Normalizing ensures the closest synapses always have an effective attenuation of 1.0.
        att = priors_dict["attenuation_matrix"].clone()
        max_att = att.max(dim=1, keepdim=True)[0]
        max_att[max_att == 0] = 1.0  # Prevent division by zero
        att = att / max_att

        self.register_buffer("attenuation", att)
        self.register_buffer("adjacency", priors_dict["adjacency_matrix"])

    def get_effective_weights(self):
        return (
            torch.abs(self.raw_weights)
            * self.dale_mask
            * self.attenuation
            * self.adjacency
        )

    def forward(self, pre_spikes, stp_state=None):
        """
        pre_spikes: (Batch, InFeatures)
        stp_state: Optional tensor (Batch, InFeatures) tracking neurotransmitter reserves [0, 1]
        Returns: (current, new_stp_state)
        """
        effective_weights = self.get_effective_weights()

        if stp_state is None:
            # If no STP state is provided, just do standard linear
            return F.linear(pre_spikes, effective_weights), None

        # Short-Term Depression (Neurotransmitter Depletion)
        # Spiking depletes resources by a utilization factor
        # Resources recover towards 1.0 with tau_d (recovery rate = 0.05)
        recovery_rate = 0.05
        
        released_transmitter = stp_state * self.U * pre_spikes

        new_stp_state = (
            stp_state + recovery_rate * (1.0 - stp_state) - released_transmitter
        )

        # Current is proportional to the released transmitter.
        # We multiply by (1.0 / self.U) so that the initial spike (when stp_state=1.0)
        # produces the exact same current magnitude as the network did before STP was added.
        current = F.linear(released_transmitter, effective_weights) * (1.0 / self.U)

        return current, new_stp_state
