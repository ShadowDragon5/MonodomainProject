import csv

import torch
import torch.nn as nn
from tqdm import tqdm

from deep_o_net import DeepONet, MonodomainPINN

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Sigma_h = 9.5298e-4
# Sigma_d_factors = [10, 1, 0.1]
a, fr, ft, fd = 18.515, 0.0, 0.2383, 1.0
T_final = 35.0
Sigma_d_factor = 10


def u0(x, y):
    return ((x >= 0.9) & (y >= 0.9)).double()


branch_dim, trunk_dim, output_dim = 3, 3, 100
deeponet = DeepONet(
    branch_dim,
    trunk_dim,
    output_dim,
    trunk_hidden_layers=[64, 64],
    branch_hidden_layers=[64, 64],
    activation=nn.Tanh(),
).to(device)

model = MonodomainPINN(deeponet, Sigma_h, Sigma_d_factor, device).to(device)

weights = torch.load("saves/model1.pth", weights_only=True)
model.load_state_dict(weights)


def make_data(N_f, N_ic, N_bc, Sigma_h, T_final, u0, device):
    x_f = torch.rand(N_f, 1, device=device, requires_grad=True)
    y_f = torch.rand(N_f, 1, device=device, requires_grad=True)
    t_f = T_final * torch.rand(N_f, 1, device=device, requires_grad=True)
    trunk_input_f = torch.cat([x_f, y_f, t_f], dim=1)
    branch_input_f = torch.tensor(
        [[Sigma_h, Sigma_h, Sigma_h]], dtype=torch.float64, device=device
    ).repeat(N_f, 1)

    x_ic = torch.rand(N_ic, 1, device=device, requires_grad=True)
    y_ic = torch.rand(N_ic, 1, device=device, requires_grad=True)
    t_ic = torch.zeros(N_ic, 1, device=device, requires_grad=True)
    trunk_input_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    branch_input_ic = torch.tensor(
        [[Sigma_h, Sigma_h, Sigma_h]], dtype=torch.float64, device=device
    ).repeat(N_ic, 1)
    u_ic_target = u0(x_ic, y_ic).to(device)

    x_bc = torch.cat(
        [
            torch.zeros(N_bc // 4, 1, device=device),
            torch.ones(N_bc // 4, 1, device=device),
            torch.rand(N_bc // 2, 1, device=device),
        ]
    ).requires_grad_(True)

    y_bc = torch.cat(
        [
            torch.rand(N_bc // 4, 1, device=device),
            torch.rand(N_bc // 4, 1, device=device),
            torch.zeros(N_bc // 4, 1, device=device),
            torch.ones(N_bc // 4, 1, device=device),
        ]
    ).requires_grad_(True)

    t_bc = T_final * torch.rand(N_bc, 1, device=device)
    trunk_input_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)
    branch_input_bc = torch.tensor(
        [[Sigma_h, Sigma_h, Sigma_h]], dtype=torch.float64, device=device
    ).repeat(N_bc, 1)

    return (
        branch_input_f,
        trunk_input_f,
        branch_input_ic,
        trunk_input_ic,
        u_ic_target,
        branch_input_bc,
        trunk_input_bc,
    )


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def get_closure(
    model,
    branch_input_f,
    trunk_input_f,
    branch_input_ic,
    trunk_input_ic,
    u_ic_target,
    branch_input_bc,
    trunk_input_bc,
):
    def closure():
        model.zero_grad()
        res = model.PDE_residual(branch_input_f, trunk_input_f)
        loss_pde = torch.mean(res**2)

        u_ic_pred = model.deeponet(branch_input_ic, trunk_input_ic)
        loss_ic = nn.MSELoss()(u_ic_pred, u_ic_target)

        loss_bc = model.BC_loss(branch_input_bc, trunk_input_bc)
        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        return loss

    return closure


def train_monodomain(
    model, epochs=500, lr=1e-5, N_f=10000, N_ic=1000, N_bc=1000, pre_epoch=100
):
    model.to(device)
    data = make_data(N_f, N_ic, N_bc, Sigma_h, T_final, u0, device)
    (
        branch_input_f,
        trunk_input_f,
        branch_input_ic,
        trunk_input_ic,
        u_ic_target,
        branch_input_bc,
        trunk_input_bc,
    ) = data

    best_loss = float("inf")
    best_state = None

    with open("loss_log.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "loss_pde", "loss_ic", "loss_bc"])

        for epoch in range(100):
            print("\n[ Layer-wise Pretraining with L-BFGS ]")
            freeze_all(model)

            for net_name in ["branch_net", "trunk_net"]:
                net = getattr(model.deeponet, net_name)
                for i, layer in enumerate(net):
                    if isinstance(layer, nn.Linear):
                        print(f"Pretraining {net_name} layer {i}")
                        unfreeze_module(layer)

                        optimizer = torch.optim.LBFGS(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            max_iter=pre_epoch,
                            history_size=50,
                            line_search_fn="strong_wolfe",
                        )

                        pbar = tqdm(
                            range(pre_epoch),
                            desc=f"Layer {i} ({net_name})",
                            leave=False,
                        )

                        def closure():
                            model.zero_grad()
                            res = model.PDE_residual(branch_input_f, trunk_input_f)
                            loss_pde = torch.mean(res**2)

                            u_ic_pred = model.deeponet(branch_input_ic, trunk_input_ic)
                            loss_ic = nn.MSELoss()(u_ic_pred, u_ic_target)

                            loss_bc = model.BC_loss(branch_input_bc, trunk_input_bc)
                            loss = loss_pde + loss_ic + loss_bc
                            loss.backward()
                            pbar.set_postfix_str(
                                f"loss={loss.item():.2e} | PDE={loss_pde.item():.2e}, IC={loss_ic.item():.2e}, BC={loss_bc.item():.2e}"
                            )
                            return loss

                        optimizer.step(closure)
                        pbar.close()

                        freeze_module(layer)

            print("\n[ Full Network Training with L-BFGS ]")
            for param in model.parameters():
                param.requires_grad = True

            pbar = tqdm(range(epochs), desc="Full Training", leave=True)
            tracker = {}

            def final_closure():
                optimizer.zero_grad()
                res = model.PDE_residual(branch_input_f, trunk_input_f)
                loss_pde = torch.mean(res**2)

                u_ic_pred = model.deeponet(branch_input_ic, trunk_input_ic)
                loss_ic = nn.MSELoss()(u_ic_pred, u_ic_target)

                loss_bc = model.BC_loss(branch_input_bc, trunk_input_bc)

                loss = loss_pde + loss_ic + loss_bc
                loss.backward()

                nonlocal best_loss, best_state
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = model.state_dict()

                pbar.set_postfix_str(
                    f"loss={loss.item():.2e} | PDE={loss_pde.item():.2e}, IC={loss_ic.item():.2e}, BC={loss_bc.item():.2e}"
                )
                pbar.update(1)

                tracker["loss_total"] = loss.item()
                tracker["loss_pde"] = loss_pde.item()
                tracker["loss_ic"] = loss_ic.item()
                tracker["loss_bc"] = loss_bc.item()

                return loss

            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=epochs,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            optimizer.step(final_closure)

            writer.writerow(
                [
                    epoch,
                    tracker["loss_total"],
                    tracker["loss_pde"],
                    tracker["loss_ic"],
                    tracker["loss_bc"],
                ]
            )

            pbar.close()

    torch.save(best_state, "saves/model.pth")


train_monodomain(
    model,
)
