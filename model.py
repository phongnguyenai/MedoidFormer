import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MedoidGenerator, MLP_Res, MedoidTransformer

# ------------------------------------------------------------
# Encoder: Extracts high-level features from input point cloud
# ------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, out_dim=1024, num_landmarks=17, k_g=512, k_t=16):
        """
        Encoder module that extracts global feature representation from partial point cloud.
        Args:
            out_dim (int): Output feature dimension.
            num_landmarks (int): Number of landmarks (not used inside encoder directly).
            k_g (int): Number of points for medoid generation grouping.
            k_t (int): Number of neighbors for medoid transformation.
        """
        super(Encoder, self).__init__()

        # Hierarchical Medoid-based feature extraction layers
        self.generator_1 = MedoidGenerator(npoint=512, nsample=k_g, in_channel=3, mlp=[64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = MedoidTransformer(128, dim=64, n_knn=k_t)
        self.generator_2 = MedoidGenerator(128, k_g, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = MedoidTransformer(256, dim=64, n_knn=k_t)
        self.generator_3 = MedoidGenerator(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud (Tensor): (B, 3, N) input partial point cloud.
        Returns:
            l3_points (Tensor): (B, out_dim, 1) global encoded feature.
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        # Stage 1: Medoid-based sampling and feature extraction
        l1_xyz, l1_points, idx1 = self.generator_1(l0_xyz, l0_points)
        l1_points = self.transformer_1(l1_points, l1_xyz)

        # Stage 2: Deeper medoid feature extraction
        l2_xyz, l2_points, idx2 = self.generator_2(l1_xyz, l1_points)
        l2_points = self.transformer_2(l2_points, l2_xyz)

        # Stage 3: Global feature aggregation
        l3_xyz, l3_points = self.generator_3(l2_xyz, l2_points)

        return l3_points


# ------------------------------------------------------------
# LandmarkGenerator: Decodes features to predict landmarks
# ------------------------------------------------------------
class LandmarkGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=17):
        """
        Predicts landmark positions from extracted features.
        Args:
            dim_feat (int): Input feature dimension.
            num_pc (int): Number of landmarks to predict.
        """
        super(LandmarkGenerator, self).__init__()

        # Seed point projection
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)

        # Residual MLPs for refinement
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)

        # Final landmark regression head
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat (Tensor): (B, dim_feat, 1) global feature input.
        Returns:
            landmarks (Tensor): (B, num_pc, 3) predicted landmarks.
        """
        x1 = self.ps(feat)  # (B, 128, num_pc)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat(1, 1, x1.size(2))], dim=1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat(1, 1, x2.size(2))], dim=1))
        landmarks = self.mlp_4(x3)
        return landmarks.permute(0, 2, 1)


# ------------------------------------------------------------
# MedoidFormer: Full model combining Encoder and LandmarkGenerator
# ------------------------------------------------------------
class MedoidFormer(nn.Module):
    def __init__(self, num_landmarks, dim, k_g=512, k_t=16):
        """
        Full MedoidFormer model that predicts landmarks from point cloud.
        Args:
            num_landmarks (int): Number of landmarks to predict.
            dim (int): Feature dimension.
            k_g (int): Medoid grouping parameter.
            k_t (int): Medoid transformer neighbor parameter.
        """
        super(MedoidFormer, self).__init__()
        self.feat_extractor = Encoder(out_dim=dim, k_g=k_g, k_t=k_t)
        self.seed_generator = LandmarkGenerator(dim_feat=dim, num_pc=num_landmarks)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud (Tensor): (B, N, 3) input point cloud.
        Returns:
            lmks (Tensor): (B, num_landmarks, 3) predicted landmark positions.
        """
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()  # (B, 3, N)
        feat = self.feat_extractor(point_cloud)
        lmks = self.seed_generator(feat)
        return lmks


# ------------------------------------------------------------
# FLOPs & Params Profiling (Optional - for debugging)
# ------------------------------------------------------------
# from thop import profile
# from thop import clever_format

# if __name__ == "__main__":
#     inp = torch.rand([8, 16384, 3]).cuda()
#     model = MedoidFormer(num_landmarks=68, dim=1024).cuda()
#     flops, params = profile(model, inputs=(inp,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print(f"FLOPs: {flops}")
#     print(f"Parameters: {params}")
