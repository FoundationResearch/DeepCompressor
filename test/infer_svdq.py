import torch
import torch.nn as nn

from nunchaku.models.linear import SVDQW4A4Linear
from model import TwoLayerMLP


def build_quantized_model(config: dict, state_dict: dict, metadata: dict, device: str | torch.device = "cuda") -> nn.Module:
    # Reconstruct architecture and replace Linear with SVDQ using ranks inferred from state_dict/metadata
    model = TwoLayerMLP(**config)

    precision = metadata.get("precision", "int4")

    def get_rank_for(name: str) -> int:
        key = f"{name}.proj_up"
        if key in state_dict:
            return state_dict[key].shape[1]
        ranks = metadata.get("rank", {})
        return int(ranks.get(name, 32))

    def replace(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                rank = get_rank_for(name)
                act_unsigned = name != "layer1"  # after ReLU
                svdq = SVDQW4A4Linear.from_linear(child, rank=rank, precision=precision, act_unsigned=act_unsigned)
                setattr(module, name, svdq)
            else:
                replace(child)

    replace(model)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def infer_svdq(ckpt_path: str = "./ckpt/mlp_demo_svdq.pt", device: str = "cuda", num_test_samples: int = 1024):
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    state_dict = ckpt["state_dict"]

    qmodel = build_quantized_model(config, state_dict, ckpt, device=device)

    torch.manual_seed(42)
    x = torch.randn(num_test_samples, config["in_features"], dtype=torch.bfloat16, device=device)
    # ground truth: sum of inputs
    y = torch.sum(x, dim=1, keepdim=True)
    print(x.shape)
    # SVDQ linear expects (B, S, C). If x is 2D (N, C), promote to (1, N, C) and squeeze back.
    if x.ndim == 2:
        x3 = x.unsqueeze(0)
        pred3 = qmodel(x3)
        pred = pred3.squeeze(0)
    else:
        pred = qmodel(x)
    loss = nn.MSELoss()(pred, y)
    print(f"[nunchaku infer] loss={loss.item():.6f}")
    for i in range(min(5, num_test_samples)):
        print(f"  Input sum: {y[i].item():.4f}, Predicted: {pred[i].item():.4f}")
    return pred, y, loss


if __name__ == "__main__":
    infer_svdq()


