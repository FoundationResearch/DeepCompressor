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
    
    # Ground truth: summation of inputs
    y_test = torch.sum(x_test, dim=1, keepdim=True)
    
    # Inference
    with torch.no_grad():
        pred = model(x_test)
        loss = nn.MSELoss()(pred, y_test)
    
    print(f"Inference loss: {loss.item():.6f}")
    print(f"Sample predictions vs ground truth:")
    for i in range(min(5, num_test_samples)):
        print(f"  Input sum: {y_test[i].item():.4f}, Predicted: {pred[i].item():.4f}")
    
    return pred, y_test, loss

if __name__ == "__main__":
    infer()
