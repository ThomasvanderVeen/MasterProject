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


class LIF_simple(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'G', 'count_refr', 'spk', 'I', 'h'])

    def __init__(self, parameters):
        super(LIF_simple, self).__init__()
        self.G_r = parameters['G_r']
        self.tau_G = parameters['tau_G']
        self.p = parameters['p']
        self.tau = parameters['tau']
        self.tau_min = parameters['tau_min']
        self.tau_plus = parameters['tau_plus']
        self.V_R = parameters['V_R']
        self.V_T = parameters['V_T']
        self.V_h = parameters['V_h']
        self.n = parameters['n']
        self.N_input = parameters['N_input']
        self.dt = parameters['dt']
        self.refrac = parameters['refrac']
        self.refr = self.refrac / self.dt
        self.state = None

    def initialize_state(self):
        self.state = None

    def forward(self, input, input_2):
        if self.state is None:
            self.state = self.NeuronState(V=torch.full((self.n,), self.V_R, dtype=torch.float64, device=input.device),
                                          G=torch.full((self.n, self.N_input), self.G_r, device=input.device),
                                          count_refr=torch.full((self.n, self.N_input), self.refr, device=input.device),
                                          spk=torch.zeros(self.n, device=input.device),
                                          I=torch.zeros((self.n), device=input.device),
                                          h=torch.zeros((self.n), device=input.device))



        V = self.state.V
        G = self.state.G
        count_refr = self.state.count_refr
        I = self.state.I
        h = self.state.h



        I = h * torch.heaviside(V - self.V_h + 1e-4, torch.zeros(self.n, dtype=torch.float64)) * (V - self.V_R)*self.dt/0.5e-4

        h += -self.dt*(h/self.tau_min)*torch.heaviside(V - self.V_h, torch.zeros(self.n, dtype=torch.float64)) + \
             (self.dt*(1-h)/self.tau_plus)*torch.heaviside(self.V_h - V, torch.zeros(self.n, dtype=torch.float64))

        #G = G * (1-input_2)

        V += self.dt * (self.V_R - V) / self.tau

        V += torch.sum(G * input * torch.heaviside(G - self.G_r*0.8, torch.zeros_like(G)), dim=1) + I




        G += self.dt * (self.G_r - G) / self.tau_G
        G += -G*(1-self.p)*input

        spk = activation(V - self.V_T)



        count_refr = self.refr * input + (1 - input) * (count_refr - 1)

        V = (1 - spk) * V + spk * (self.V_h)
        #G = (1 - input) * self.G_r * (count_refr <= 0) + input * 0 + (1 - input) * 0 * (count_refr > 0)

        self.state = self.NeuronState(V=V, G=G, count_refr=count_refr, spk=spk, I=I, h=h)

        return V, spk