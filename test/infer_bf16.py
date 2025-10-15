import torch
import torch.nn as nn
from model import TwoLayerMLP

def infer(ckpt_path="./ckpt/mlp_demo.pt", device="cuda", num_test_samples=1024):
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    
    # Create model and load weights
    model = TwoLayerMLP(**config).to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Generate test data (same as training)
    torch.manual_seed(42)
    x_test = torch.randn(num_test_samples, config["in_features"], dtype=torch.bfloat16, device=device)
    
    # Ground truth: 16-d; [sum(x), -sum(x), 0, ..., 0]
    s = torch.sum(x_test, dim=1, keepdim=True)
    y_test = torch.zeros(num_test_samples, config["out_features"], dtype=torch.bfloat16, device=device)
    y_test[:, 0:1] = s
    y_test[:, 1:2] = -s
    
    # Inference
    with torch.no_grad():
        pred = model(x_test)
        loss = nn.MSELoss()(pred, y_test)
    
    print(f"Inference loss (BF16): {loss.item():.6f}")
    print("Sample [y0, y1] vs GT [sum, -sum]:")
    for i in range(min(5, num_test_samples)):
        print(f"  GT: [{y_test[i,0].item():.4f}, {y_test[i,1].item():.4f}] | Pred: [{pred[i,0].item():.4f}, {pred[i,1].item():.4f}]")
    
    return pred, y_test, loss

if __name__ == "__main__":
    infer()
