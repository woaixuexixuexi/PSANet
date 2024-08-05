import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Channel Attention Module
class CABlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(CABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.eca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, image_features):
        feature1 = self.conv1(image_features)
        feature2 = self.conv2(image_features)
        feature3 = self.conv3(image_features)
        features = feature1 + feature2 + feature3
        features = self.dropout(features)
        y = self.avg_pool(features)
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        avg_pool_weights = self.eca(y)
        weights = torch.sigmoid(avg_pool_weights)
        weights = weights.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return features * weights, weights

# Proto Channel Attention Block
class PCABlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(PCABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.eca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, image_features, proto):
        feature1 = self.conv1(image_features)
        feature2 = self.conv2(image_features)
        feature3 = self.conv3(image_features)
        features = feature1 + feature2 + feature3
        features = self.dropout(features)
        avg_pooled_image_features = self.avg_pool(features)

        y = avg_pooled_image_features + proto
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        avg_pool_weights = self.eca(y)
        weights = torch.sigmoid(avg_pool_weights)
        weights = weights.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1

        return features * weights, weights

# Spatial Attention Module
class SABlock(nn.Module):
    def __init__(self):
        super(SABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        in_channels = 256
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features):
        feature1 = self.conv1(image_features)
        feature2 = self.conv2(image_features)
        feature3 = self.conv3(image_features)
        transpose_features1 = feature1.view(*feature1.shape[:2], -1).transpose(1, 2)#torch.Size([8, 900, 256])
        transpose_features2 = feature2.view(*feature2.shape[:2], -1).transpose(1, 2)
        transpose_features3 = feature3.view(*feature3.shape[:2], -1).transpose(1, 2)
        avg_pooled_features1 = self.avg_pool(transpose_features1)#torch.Size([8, 900, 1])
        max_pooled_features1 = self.max_pool(transpose_features1)
        avg_pooled_features2 = self.avg_pool(transpose_features2)  # torch.Size([8, 900, 1])
        max_pooled_features2 = self.max_pool(transpose_features2)
        avg_pooled_features3 = self.avg_pool(transpose_features3)  # torch.Size([8, 900, 1])
        max_pooled_features3 = self.max_pool(transpose_features3)
        pooled_features1 = torch.cat((avg_pooled_features1, max_pooled_features1), 2)
        pooled_features1 = pooled_features1.transpose(1, 2).view(-1, 2, *image_features.shape[2:])#torch.Size([8, 2, 30, 30])
        pooled_features2 = torch.cat((avg_pooled_features2, max_pooled_features2), 2)
        pooled_features2 = pooled_features2.transpose(1, 2).view(-1, 2, *image_features.shape[2:])#torch.Size([8, 2, 30, 30])
        pooled_features3 = torch.cat((avg_pooled_features3, max_pooled_features3), 2)
        pooled_features3 = pooled_features3.transpose(1, 2).view(-1, 2, *image_features.shape[2:])#torch.Size([8, 2, 30, 30])
        weights1 = torch.sigmoid(self.conv(pooled_features1))
        weights2 = torch.sigmoid(self.conv(pooled_features2))
        weights3 = torch.sigmoid(self.conv(pooled_features3))
        weights = weights1 + weights2 + weights3
        weights = torch.sigmoid(weights)

        return image_features * weights, weights

# Proto Spatial Attention Module
class PSABlock(nn.Module):
    def __init__(self):
        super(PSABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        in_channels = 256
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels*2, in_channels, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features, proto):
        features = torch.cat(
            (image_features, proto.expand(*proto.shape[:2], *image_features.shape[2:])), 1)  # broadcast along height and width dimension ([8, 512, 30, 30])
        feature1 = self.conv1(features)
        feature2 = self.conv2(features)
        feature3 = self.conv3(features)
        transpose_features1 = feature1.view(*feature1.shape[:2], -1).transpose(1, 2)  # torch.Size([8, 900, 256])
        transpose_features2 = feature2.view(*feature2.shape[:2], -1).transpose(1, 2)
        transpose_features3 = feature3.view(*feature3.shape[:2], -1).transpose(1, 2)
        avg_pooled_features1 = self.avg_pool(transpose_features1)  # torch.Size([8, 900, 1])
        max_pooled_features1 = self.max_pool(transpose_features1)
        avg_pooled_features2 = self.avg_pool(transpose_features2)  # torch.Size([8, 900, 1])
        max_pooled_features2 = self.max_pool(transpose_features2)
        avg_pooled_features3 = self.avg_pool(transpose_features3)  # torch.Size([8, 900, 1])
        max_pooled_features3 = self.max_pool(transpose_features3)
        pooled_features1 = torch.cat((avg_pooled_features1, max_pooled_features1), 2)
        pooled_features1 = pooled_features1.transpose(1, 2).view(-1, 2, *image_features.shape[2:])  # torch.Size([8, 2, 30, 30])
        pooled_features2 = torch.cat((avg_pooled_features2, max_pooled_features2), 2)
        pooled_features2 = pooled_features2.transpose(1, 2).view(-1, 2, *image_features.shape[2:])  # torch.Size([8, 2, 30, 30])
        pooled_features3 = torch.cat((avg_pooled_features3, max_pooled_features3), 2)
        pooled_features3 = pooled_features3.transpose(1, 2).view(-1, 2, *image_features.shape[2:])  # torch.Size([8, 2, 30, 30])
        weights1 = torch.sigmoid(self.conv(pooled_features1))
        weights2 = torch.sigmoid(self.conv(pooled_features2))
        weights3 = torch.sigmoid(self.conv(pooled_features3))
        weights = weights1 + weights2 + weights3
        weights = torch.sigmoid(weights)

        return image_features * weights, weights

def get_addition_loss(ca_weights, pca_weights, sa_weights, psa_weights, cho):
    if cho == 'norm_softmargin':
        total_loss = 1 * F.soft_margin_loss(ca_weights / torch.norm(ca_weights, p=2, dim=1, keepdim=True), pca_weights.detach() / torch.norm(pca_weights.detach(), p=2, dim=1, keepdim=True))
        total_loss += 0.1 * F.soft_margin_loss(sa_weights / torch.norm(sa_weights, p=2, dim=(2,3), keepdim=True), psa_weights.detach() / torch.norm(psa_weights.detach(), p=2, dim=(2,3), keepdim=True))
    elif cho == 'softmargin':
        total_loss = 1 * F.soft_margin_loss(ca_weights, pca_weights.detach())
        total_loss += 0.1 * F.soft_margin_loss(sa_weights, psa_weights.detach())
    return total_loss

class PSAM(nn.Module):
    def __init__(self, reduce_dim):
        super(PSAM, self).__init__()
        self.ca_block = CABlock(reduce_dim)
        self.pca_block = PCABlock(reduce_dim)
        self.sa_block = SABlock()
        self.psa_block = PSABlock()
    def forward(self, fearure, proto=None, Support=False):
        if Support:    # proto-guided
            ca_supp_feat, ca_s = self.ca_block(fearure)
            pca_supp_feat, pca_s = self.pca_block(fearure, proto)
            supp_feat1 = pca_supp_feat
            sa_supp_feat, sa_s = self.sa_block(supp_feat1)
            psa_supp_feat, psa_s = self.psa_block(supp_feat1, proto)
            return psa_supp_feat, ca_s, pca_s, sa_s, psa_s
        else:    # self-guided
            ca_query_feat, ca_q = self.ca_block(fearure)
            query_feat1 = ca_query_feat
            sa_query_feat, sa_q = self.sa_block(query_feat1)
            return sa_query_feat, ca_q, sa_q