import torch
import torch.nn as nn
import math


class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, potential, threshold, slope=25.0):
        ctx.save_for_backward(potential, threshold)
        ctx.slope = slope
        return (potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        potential, threshold = ctx.saved_tensors
        grad_p = 1.0 / (1.0 + ctx.slope * torch.abs(potential - threshold)) ** 2
        return grad_output * grad_p, None, None


class BioPerceptronNeuron(nn.Module):
    def __init__(
        self, in_features, out_features, priors_dict, tau_m=20.0, tau_theta=50.0
    ):
        super().__init__()
        self.out_features = out_features
        self.alpha = math.exp(-1.0 / tau_m)
        self.decay_theta = math.exp(-1.0 / tau_theta)

        self.noise_std = 0.02
        self.gamma_glia = 0.02
        self.theta_0 = 0.5
        self.theta_exc = 0.05
        self.theta_inh = 0.05

        self.register_buffer("is_exc", priors_dict["post_is_exc"])
        # Synapse needs to be set by the model wrapper
        self.synapse = None

    def forward(self, input_sequence, return_states=False):
        batch_size, time_steps, _ = input_sequence.shape
        device = input_sequence.device

        membrane_potential = torch.zeros(batch_size, self.out_features, device=device)
        dynamic_threshold = (
            torch.ones(batch_size, self.out_features, device=device) * self.theta_0
        )
        spikes = []
        potentials_hist = []
        thresholds_hist = []
        stp_hist = []

        stp_state = torch.ones(
            batch_size, input_sequence.size(-1), device=device
        )  # Neurotransmitter reserves start at 100%

        spike_t = torch.zeros_like(membrane_potential)

        for t in range(time_steps):
            current, stp_state = self.synapse(input_sequence[:, t, :], stp_state)

            # Dale's Law Lateral Current
            exc_spikes = (spike_t * self.is_exc).sum(dim=1, keepdim=True)
            inh_spikes = (spike_t * ~self.is_exc).sum(dim=1, keepdim=True)
            lateral_current = self.theta_exc * (
                exc_spikes - spike_t * self.is_exc
            ) - self.theta_inh * (inh_spikes - spike_t * ~self.is_exc)

            noise_scale = math.sqrt(1 - self.alpha**2)
            noise = torch.randn_like(membrane_potential) * self.noise_std * noise_scale

            membrane_potential = (
                self.alpha * membrane_potential
                + (1 - self.alpha) * (current + lateral_current)
                + noise
            )

            if return_states:
                potentials_hist.append(membrane_potential.clone())
                thresholds_hist.append(dynamic_threshold.clone())
                stp_hist.append(stp_state.clone())

            spike_t = FastSigmoidSurrogate.apply(membrane_potential, dynamic_threshold)
            spikes.append(spike_t)

            dynamic_threshold = (
                self.theta_0
                + (dynamic_threshold - self.theta_0) * self.decay_theta
                + self.gamma_glia * spike_t
            ).detach()

            membrane_potential = membrane_potential * (1.0 - spike_t.detach())

        spikes_tensor = torch.stack(spikes, dim=1)
        if return_states:
            return spikes_tensor, {
                "potentials": torch.stack(potentials_hist, dim=1),
                "thresholds": torch.stack(thresholds_hist, dim=1),
                "stp_states": torch.stack(stp_hist, dim=1),
            }
        return spikes_tensor
