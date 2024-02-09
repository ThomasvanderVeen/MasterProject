from collections import namedtuple

import torch
import torch.nn as nn


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 20.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


activation = SurrGradSpike.apply


class AdEx(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'w', 'count_refr', 'spk'])

    def __init__(self, parameters):
        super(AdEx, self).__init__()
        self.C = parameters['C']
        self.g_L = parameters['g_L']
        self.E_L = parameters['E_L']
        self.DeltaT = parameters['DeltaT']
        self.a = parameters['a']
        self.V_T = parameters['V_T']
        self.tau_W = parameters['tau_W']
        self.b = parameters['b']
        self.V_R = parameters['V_R']
        self.V_cut = parameters['V_cut']
        self.n = parameters['n']
        self.dt = parameters['dt']
        self.refrac = parameters['refrac']
        self.refr = self.refrac / self.dt
        self.state = None

    def initialize_state(self):
        self.state = None

    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.linspace(self.E_L, self.E_L+10e-3, self.n, device=input.device),
                                          w=torch.linspace(0, 0, self.n, device=input.device),
                                          count_refr=torch.zeros(self.n, device=input.device),
                                          spk=torch.zeros(self.n, device=input.device))
        V = self.state.V
        w = self.state.w
        I = input
        count_refr = self.state.count_refr

        V += (self.g_L * (self.E_L - V) + self.g_L * self.DeltaT * torch.exp(
            (V - self.V_T) / self.DeltaT) - w + I) * self.dt / self.C

        spk = activation(V - self.V_cut)
        count_refr = self.refr * spk + (1 - spk) * (count_refr - 1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.V_R + (1 - spk) * self.V_R * (count_refr > 0)
        w += spk * self.b
        w += (self.a * (V - self.E_L) - w) * self.dt / self.tau_W

        self.state = self.NeuronState(V=V, w=w, count_refr=count_refr, spk=spk)

        return V, spk
