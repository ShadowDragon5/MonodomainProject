import torch
import torch.nn as nn

# Sigma_h = 9.5298e-4
# Sigma_d_factors = [10, 10, 10]
a, fr, ft, fd = 18.515, 0.0, 0.2383, 1.0
# T_final = 35.0


class DeepONet(nn.Module):
    def __init__(
        self,
        branch_dim,
        trunk_dim,
        output_dim,
        trunk_hidden_layers,
        branch_hidden_layers,
        activation,
    ):
        super(DeepONet, self).__init__()

        layers_branch = []
        input_dim = branch_dim
        for hidden_dim in branch_hidden_layers:
            layers_branch.append(nn.Linear(input_dim, hidden_dim))
            layers_branch.append(activation)
            input_dim = hidden_dim
        layers_branch.append(nn.Linear(input_dim, output_dim))
        self.branch_net = nn.Sequential(*layers_branch)

        layers_trunk = []
        input_dim = trunk_dim
        for hidden_dim in trunk_hidden_layers:
            layers_trunk.append(nn.Linear(input_dim, hidden_dim))
            layers_trunk.append(activation)
            input_dim = hidden_dim
        layers_trunk.append(nn.Linear(input_dim, output_dim))
        self.trunk_net = nn.Sequential(*layers_trunk)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)


class MonodomainPINN(nn.Module):
    def __init__(self, deeponet, Sigma_h, Sigma_d_factor, device):
        super(MonodomainPINN, self).__init__()
        self.deeponet = deeponet
        self.Sigma_h = Sigma_h
        self.Sigma_d = Sigma_h * Sigma_d_factor
        self.a, self.fr, self.ft, self.fd = a, fr, ft, fd
        self.device = device

    def f(self, u):
        return self.a * (u - self.fr) * (u - self.ft) * (u - self.fd)

    def sigma(self, x, y):
        diseased_regions = [((0.3, 0.7), 0.1), ((0.7, 0.3), 0.15), ((0.5, 0.5), 0.1)]
        sigma = torch.full_like(x, self.Sigma_h)
        for (cx, cy), radius in diseased_regions:
            mask = (x - cx) ** 2 + (y - cy) ** 2 < radius**2
            sigma[mask] = self.Sigma_d
        return sigma

    def PDE_residual(self, branch_input, trunk_input):
        x = trunk_input[:, 0:1].clone().detach().requires_grad_(True)
        y = trunk_input[:, 1:2].clone().detach().requires_grad_(True)
        t = trunk_input[:, 2:3].clone().detach().requires_grad_(True)

        inputs = torch.cat([x, y, t], dim=1)
        u = self.deeponet(branch_input, inputs)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            u_y,
            y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True,
        )[0]

        sigma = self.sigma(x, y)

        residual = u_t - sigma * (u_xx + u_yy) + self.f(u)
        return residual

    def IC_loss(self, branch_input, trunk_input, u0):
        u_pred = self.deeponet(branch_input, trunk_input)
        return nn.MSELoss()(u_pred, u0)

    def BC_loss(self, branch_input, trunk_input):
        x = trunk_input[:, 0:1].clone().detach().requires_grad_(True)
        y = trunk_input[:, 1:2].clone().detach().requires_grad_(True)
        t = trunk_input[:, 2:3].clone().detach()

        inp = torch.cat([x, y, t], dim=1)

        u = self.deeponet(branch_input, inp)

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        mask = (x == 0) | (x == 1) | (y == 0) | (y == 1)
        bc_loss = (u_x[mask] ** 2 + u_y[mask] ** 2).mean()

        return bc_loss
