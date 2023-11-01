import torch
import torch.nn as nn
from collections import namedtuple


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


class LIF_primitive(nn.Module):
    NeuronState = namedtuple('NeuronState', ['V', 'w', 'count_refr', 'spk', 'I'])

    def __init__(self, parameters):
        super(LIF_primitive, self).__init__()
        self.tau = parameters['tau']
        self.V_R = parameters['V_R']
        self.V_T = parameters['V_T']
        self.w = parameters['w']
        self.n = parameters['n']
        self.N_input = parameters['N_input']
        self.dt = parameters['dt']
        self.refrac = parameters['refrac']
        self.refr = self.refrac / self.dt
        self.state = None

    def initialize_state(self):
        self.state = None

    def forward(self, input):
        if self.state is None:
            self.state = self.NeuronState(V=torch.full((self.n,), self.V_R, dtype=torch.float64, device=input.device),
                                          w=torch.tensor(self.w, device=input.device),
                                          count_refr=torch.zeros(self.n, device=input.device),
                                          spk=torch.zeros(self.n, device=input.device),
                                          I=torch.zeros((self.n, self.N_input), device=input.device))
        V = self.state.V
        w = self.state.w
        count_refr = self.state.count_refr
        I = self.state.I

        V += self.dt*(self.V_R-V)/self.tau
        V += torch.sum(w*input, dim=-1)

        spk = activation(V - self.V_T)
        count_refr = self.refr*spk + (1-spk)*(count_refr-1)
        V = (1 - spk) * V * (count_refr <= 0) + spk * self.V_R + (1 - spk) * self.V_R * (count_refr > 0)

        self.state = self.NeuronState(V=V, w=w, count_refr=count_refr, spk=spk, I=I)

        return V, spk
