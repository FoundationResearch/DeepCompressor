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
    ranks_cfg = {"layer1": 128, "layer2": 128}
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
    print(f"[quantize] Forward OK: pred shape={tuple(pred.shape)} in {(time.time()-t0):.3f}s")
    return ckpt_out


if __name__ == "__main__":
    quantize_and_save()


