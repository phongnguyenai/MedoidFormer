from torch.utils.data import Dataset
import open3d as o3d
import os
import torch
from os import listdir
from os.path import isfile, join
import numpy as np

class LandmarkDataset(Dataset):
    def __init__(self, path):
        self.pc_dir = os.path.join(path, "pointclouds")
        self.lmk_dir = os.path.join(path, "landmarks")

        self.files = [f[:-4] for f in listdir(self.pc_dir) if isfile(join(self.pc_dir, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        pc_file = os.path.join(self.pc_dir, file+".ply")
        lmk_file = os.path.join(self.lmk_dir, file+".npy")
        pcd = o3d.io.read_point_cloud(pc_file)

        pcd_np = np.asarray(pcd.points)
        pcd_torch = torch.Tensor(pcd_np)

        lmk_np = np.load(lmk_file)
        lmk_torch = torch.Tensor(lmk_np)

        return pcd_torch, lmk_torch, file