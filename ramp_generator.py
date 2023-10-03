import torch


class RampGenerator:
    def __init__(self, parameters_ramp, device=None):
        self.n_ramp = parameters_ramp['n_ramp']
        self.n_steps = parameters_ramp['n_steps']
        self.height = parameters_ramp['height']
        self.low = parameters_ramp['low']
        self.device = device

    def ramp(self):
        n_dims = self.n_ramp.size

        input_ramp = torch.empty((self.n_steps, n_dims))

        for i in range(n_dims):
            t1 = torch.linspace(self.low[i], self.height[i], steps=self.n_ramp[i], dtype=torch.float64, device=self.device)
            t2 = torch.linspace(self.height[i], self.height[i], steps=self.n_steps - 2 * self.n_ramp[i],
                                dtype=torch.float64, device=self.device)
            t3 = torch.linspace(self.height[i], self.low[i], steps=self.n_ramp[i], dtype=torch.float64, device=self.device)

            input_ramp[:, i] = torch.cat((t1, t2, t3))

        return input_ramp
