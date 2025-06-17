import argparse
import sys
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from dataset import LandmarkDataset
from model import MedoidFormer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# Utility class for logging outputs to multiple destinations (console, file, etc.)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass


# Euclidean loss function: Mean distance between predicted and ground truth landmarks
def euclidean_loss(gt, pred):
    distance = torch.norm(pred - gt, dim=2)
    return distance.mean()


# Training loop for one epoch
def train():
    model.train()
    train_loss = 0
    count = 0

    for data in train_dataloader:
        inp, gt = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        pred = model(inp)
        loss = euclidean_loss(gt, pred)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1
        print(f"{model_name} - Training epoch {epoch}: {int(count/len(train_dataloader)*100)}%", end='\r')
        sys.stdout.flush()

    return train_loss / len(train_dataset)


# Validation loop
def validate():
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in val_dataloader:
            inp, gt = data[0].to(device), data[1].to(device)
            pred = model(inp)
            loss = euclidean_loss(gt, pred)
            val_loss += loss.item()

    return val_loss / len(val_dataset)


# Testing loop (only runs after training completes)
def test():
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data in test_dataloader:
            inp, gt = data[0].to(device), data[1].to(device)
            pred = model(inp)
            loss = euclidean_loss(gt, pred)
            test_loss += loss.item()

    return test_loss / len(test_dataset)


# Learning rate warmup schedule
def warmup_lambda(epoch):
    if epoch < args.warmup_epochs:
        return float(epoch + 1) / float(args.warmup_epochs)
    return 1.0


# Main entry point
if __name__ == "__main__":
    torch.manual_seed(42)

    # Argument parser for configurable hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100000, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=100, help="Warmup epochs for learning rate")
    parser.add_argument("--num-lmk", type=int, default=68, help="Number of landmarks (68 for Facescape, 17 for ALD/UOW3D)")
    parser.add_argument("--model-name", default="MedoidFormer", help="Model name for saving checkpoints")
    parser.add_argument("--dataset-path", default="datasets/ALD/", help="Path to dataset")
    args = parser.parse_args()

    # Device configuration (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name
    max_patience = 500

    # Model, optimizer, and schedulers initialization
    model = MedoidFormer(num_landmarks=int(args.num_lmk), dim=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

    # Dataset preparation and splitting
    dataset = LandmarkDataset(path=str(args.dataset_path))
    train_dataset, test_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    val_dataset, test_dataset = torch.utils.data.random_split(test_val, [0.3, 0.7])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # Directory for saving model checkpoints
    os.makedirs(f"pretrained/{model_name}", exist_ok=True)

    min_val_loss = np.inf
    patience = 0

    # Main training loop
    for epoch in range(args.epochs):
        train_loss = train()
        val_loss = validate()

        # Scheduler update
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        # Early stopping & checkpoint saving
        if val_loss < min_val_loss:
            patience = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), f"pretrained/{model_name}/best.pt")
        else:
            patience += 1
            if patience > max_patience:
                break

        print(f"Epoch {epoch:06d}, Train loss: {train_loss:.10f}, Validation loss: {val_loss:.10f}, Min val loss: {min_val_loss:.10f}, Patience: {patience}")
        torch.save(model.state_dict(), f"pretrained/{model_name}/last.pt")

    # Final testing after training
    test_loss = test()
    print(f"Test loss: {test_loss:.10f}")
