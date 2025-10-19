import os
import sys
import time
import torch
import torch.nn as nn

from model import TwoLayerMLP
from SVDQuantLinearManual import SVDQuantLinearManual

def quantize_and_save(
    ckpt_in: str = "./ckpt/mlp_demo.pt",
    ckpt_out: str = "./ckpt/mlp_demo_svdq.pt",
    device: str | torch.device = "cuda",
):
    t0 = time.time()
    print("[quantize] === Start ===")
    ckpt = torch.load(ckpt_in, map_location=device)
    cfg = ckpt["config"]

    print("[quantize] Build BF16 model & load weights")
    model = TwoLayerMLP(**cfg).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["state_dict"])  # trained bf16 weights
    model.eval()

    # Quantize full model using our manual class
    ranks_cfg = {"layer1": 32, "layer2": 32}
    qmodel = SVDQuantLinearManual.quantize_model(model, ranks=ranks_cfg, device=device)

    # Optional save
    if ckpt_out:
        os.makedirs(os.path.dirname(ckpt_out) or ".", exist_ok=True)
        torch.save({"config": cfg, "state_dict": qmodel.state_dict(), "svdq_manual": True}, ckpt_out)
        print(f"[quantize] Saved checkpoint to {ckpt_out}")

    # Quick sanity forward
    torch.manual_seed(42)
    x = torch.randn(1024, cfg["in_features"], dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        pred = qmodel(x)
    # Build ground-truth as in quantize-bkup.py
    if cfg.get("out_features", 1) == 1:
        y = torch.sum(x, dim=1, keepdim=True)
    else:
        s = torch.sum(x, dim=1, keepdim=True)
        y = torch.zeros(x.shape[0], cfg["out_features"], dtype=torch.bfloat16, device=device)
        y[:, 0:1] = s
        if cfg["out_features"] > 1:
            y[:, 1:2] = -s
    loss = nn.MSELoss()(pred, y)
    print(f"[quantize] Direct-infer loss={loss.item():.6f}")
    num_show = min(5, pred.shape[0])
    for i in range(num_show):
        if cfg.get("out_features", 1) == 1:
            print(f"  GT sum={y[i,0].item():.4f} | Pred={pred[i,0].item():.4f}")
        else:
            print(
                f"  GT: [{y[i,0].item():.4f}, {y[i,1].item():.4f}] | Pred: [{pred[i,0].item():.4f}, {pred[i,1].item():.4f}]"
            )
    print(f"[quantize] Forward OK: pred shape={tuple(pred.shape)} in {(time.time()-t0):.3f}s")
    return ckpt_out


if __name__ == "__main__":
    quantize_and_save()


