import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import LandmarkDataset
from model import MedoidFormer

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-lmk", type=int, default=68, help="Number of landmarks")
    parser.add_argument("--dataset-path", default="datasets/ALD/", help="Dataset path")
    parser.add_argument("--weights-path", default="pretrained/MedoidFormer/best.pt", help="Path to model weights")
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and weights
    model = MedoidFormer(num_landmarks=args.num_lmk, dim=1024).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = LandmarkDataset(path=args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Inference loop
    for idx, (inp, gt) in enumerate(dataloader):
        inp = inp.to(device)
        pred = model(inp)  # (B, num_lmk, 3)
        pred_np = pred.cpu().detach().numpy()

        print(f"Sample {idx}:")
        print(pred_np[0])  # Print predicted landmarks for each sample

        # Optionally save output here if needed
        # np.save(f"output/sample_{idx}.npy", pred_np[0])
