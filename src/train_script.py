import torch
import torch.nn as nn
import torch.autograd as ag
import numpy as np
import matplotlib.pyplot as plt
from deep_o_net import DeepONet
from deep_o_net import MonodomainPINN
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Sigma_h = 9.5298e-4
# Sigma_d_factors = [10, 1, 0.1]
a, fr, ft, fd = 18.515, 0.0, 0.2383, 1.0
T_final = 35.0
Sigma_d_factor = 10  

def u0(x, y):
    return ((x >= 0.9) & (y >= 0.9)).double()

branch_dim, trunk_dim, output_dim = 3, 3, 100
deeponet = DeepONet(
    branch_dim, trunk_dim, output_dim,
    trunk_hidden_layers=[64, 64],
    branch_hidden_layers=[64, 64],
    activation=nn.Tanh()
).to(device)

model = MonodomainPINN(deeponet, Sigma_h, Sigma_d_factor, device).to(device)

def train_monodomain(model, epochs=5000, lr=1e-3, N_f=10000, N_ic=1000, N_bc=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_f = torch.rand(N_f, 1, device=device, requires_grad=True )
    y_f = torch.rand(N_f, 1, device=device, requires_grad=True)
    t_f = T_final * torch.rand(N_f, 1, device=device, requires_grad=True)
    trunk_input_f = torch.cat([x_f, y_f, t_f], dim=1)
    branch_input_f = torch.tensor([[Sigma_h, Sigma_h, Sigma_h]], device=device).repeat(N_f, 1)

    x_ic = torch.rand(N_ic, 1, device=device, requires_grad=True)
    y_ic = torch.rand(N_ic, 1, device=device, requires_grad=True)
    t_ic = torch.zeros(N_ic, 1, device=device, requires_grad=True)
    trunk_input_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    branch_input_ic = torch.tensor([[Sigma_h, Sigma_h, Sigma_h]], device=device).repeat(N_ic, 1)
    u_ic_target = u0(x_ic, y_ic).to(device)

    x_bc = torch.cat([torch.zeros(N_bc//4,1), torch.ones(N_bc//4,1), torch.rand(N_bc//2,1)]).to(device).requires_grad_(True)
    y_bc = torch.cat([torch.rand(N_bc//4,1), torch.rand(N_bc//4,1), torch.zeros(N_bc//4,1), torch.ones(N_bc//4,1)]).to(device).requires_grad_(True)
    t_bc = T_final * torch.rand(N_bc, 1, device=device)
    trunk_input_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
    branch_input_bc = torch.tensor([[Sigma_h, Sigma_h, Sigma_h]], device=device).repeat(N_bc, 1)

    best_loss = np.inf
    best_state = None

    for ep in range(epochs):
        opt.zero_grad()

        res = model.PDE_residual(branch_input_f, trunk_input_f)
        loss_pde = torch.mean(res**2)

        u_ic_pred = model.deeponet(branch_input_ic, trunk_input_ic)
        loss_ic = nn.MSELoss()(u_ic_pred, u_ic_target)

        loss_bc = model.BC_loss(branch_input_bc, trunk_input_bc)

        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()

        if (ep) % 500 == 0:
            print(f"Epoch {ep+1:5d} | total-loss {loss.item():.3e} (PDE: {loss_pde.item():.3e}, IC: {loss_ic.item():.3e}, BC: {loss_bc.item():.3e})")

    model.load_state_dict(best_state)
    return model
