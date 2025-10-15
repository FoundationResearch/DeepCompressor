import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from model import TwoLayerMLP

def train_and_save(
    in_features: int = 128,
    hidden_features: int = 128,
    out_features: int = 16,
    num_samples: int = 4096,
    num_steps: int = 400,
    learning_rate: float = 1e-2,
    weight_seed: int = 42,
    device: str | torch.device = "cuda",
    ckpt_dir: str = "./ckpt",
    ckpt_name: str = "mlp_demo.pt",
) -> str:
    torch.manual_seed(weight_seed)

    # Dataset: x ~ N(0, I)
    x_train = torch.randn(num_samples, in_features, dtype=torch.bfloat16, device=device)
    # Targets: y[:,0]=sum(x); y[:,1]=-sum(x); others zeros
    s = torch.sum(x_train, dim=1, keepdim=True)
    y_train = torch.zeros(num_samples, out_features, dtype=torch.bfloat16, device=device)
    y_train[:, 0:1] = s
    y_train[:, 1:2] = -s

    model = TwoLayerMLP(in_features, hidden_features, out_features).to(device).to(torch.bfloat16)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # already on device

    model.train()
    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1 or step == num_steps:
            print(f"[train] step={step:04d} loss={loss.item():.6f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "in_features": in_features,
                "hidden_features": hidden_features,
                "out_features": out_features,
                "bias": True,
            },
            "train_meta": {
                "num_samples": num_samples,
                "num_steps": num_steps,
                "learning_rate": learning_rate,
                "seed": weight_seed,
            },
        },
        ckpt_path,
    )

    print(f"[train] checkpoint saved to {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    train_and_save()


