import os
import sys
import torch
import torch.nn as nn

from model import TwoLayerMLP
from SVDQuantLinearManual import SVDQuantLinearManual


def run_infer(
    ckpt_path: str = "./ckpt/mlp_demo_svdq.pt",
    device: str | torch.device = "cuda",
):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    # Rebuild model and load quantized state
    model = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16).eval()
    qmodel = SVDQuantLinearManual.quantize_model(model, ranks={"layer1": 32, "layer2": 32}, device=device)
    qmodel.load_state_dict(ckpt["state_dict"], strict=False)
    qmodel.eval()

    # Generate inputs and GT like training target
    torch.manual_seed(42)
    x = torch.randn(1024, cfg["in_features"], dtype=torch.bfloat16, device=device)
    if cfg.get("out_features", 1) == 1:
        y = torch.sum(x, dim=1, keepdim=True)
    else:
        s = torch.sum(x, dim=1, keepdim=True)
        y = torch.zeros(x.shape[0], cfg["out_features"], dtype=torch.bfloat16, device=device)
        y[:, 0:1] = s
        if cfg["out_features"] > 1:
            y[:, 1:2] = -s

    with torch.inference_mode():
        pred = qmodel(x)

    loss = nn.MSELoss()(pred, y)
    print(f"[infer] loss={loss.item():.6f}, pred shape={tuple(pred.shape)}")
    for i in range(min(5, pred.shape[0])):
        if cfg.get("out_features", 1) == 1:
            print(f"  GT sum={y[i,0].item():.4f} | Pred={pred[i,0].item():.4f}")
        else:
            print(
                f"  GT: [{y[i,0].item():.4f}, {y[i,1].item():.4f}] | Pred: [{pred[i,0].item():.4f}, {pred[i,1].item():.4f}]"
            )


if __name__ == "__main__":
    run_infer()


