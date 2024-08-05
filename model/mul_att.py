import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Mul_Att(nn.Module):
    def __init__(self, reduce_dim=256,pyramid_bins=[1.0, 0.5, 0.25, 0.125],shot=1):
        super(Mul_Att, self).__init__()
        self.pyramid_bins = pyramid_bins
        self.shot = shot
        classes = 2
        mask_add_num = 1

        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*(2 + self.shot)+mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),#
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)

        self.alpha_conv = [] # upsample conv
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

        self.att = Feature_Attention(in_channel=reduce_dim, split=4)

    def forward(self, query_feat, proto_v, corr_query_mask, supp_feats):
        out_list = []
        pyramid_feat_list = []
        # multi-scale loop
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            supp_feats_bin_list = []
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
                for k in range(self.shot):
                    supp_feats_bin_list.append(nn.AdaptiveAvgPool2d(bin)(supp_feats[k]))
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
                for k in range(self.shot):
                    supp_feats_bin_list.append(self.avgpool_list[idx](supp_feats[k]))

            supp_feat_bin = proto_v.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            # concat features
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)

            for k in range(self.shot):
                merge_feat_bin = torch.cat([merge_feat_bin, supp_feats_bin_list[k]], 1)

            # fusion M module
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)
            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            # resize
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            # multi-scale feature
            out_list.append(inner_out_bin) 

        # fusion multi-scale
        pyramid_feat_list = self.att(pyramid_feat_list)
        fin_feat = torch.cat(pyramid_feat_list, 1)
        return fin_feat, out_list

class Feature_Attention(nn.Module):
    def __init__(self, in_channel=256, split=4):
        super(Feature_Attention, self).__init__()
        self.s = split
        self.in_channel = in_channel
        self.value = []
        self.key = []
        self.query = []
        for i in range(self.s):
            self.value.append(nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0)
            ))
            self.key.append(nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0)
            ))
            self.query.append(nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1, stride=1, padding=0)
            ))
        self.value = nn.ModuleList(self.value)
        self.key = nn.ModuleList(self.key)
        self.query = nn.ModuleList(self.query)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, features):
        out_list = []
        b, c, h, w = features[0].shape
        for idx in range(self.s):
            feature = features[idx]
            v = self.value[idx](feature)  # torch.Size([8, 256, 30, 30])

            k = self.key[idx](feature).view(b, self.in_channel, -1)  # torch.Size([8, 256, 900])

            q = self.query[idx](feature).view(b, self.in_channel, -1)  # torch.Size([8, 256, 900])
            q = q.permute(0, 2, 1)  # torch.Size([8, 900, 256])

            att_m = torch.matmul(q, k)  # torch.Size([8, 900, 900])
            pool_m = self.avg_pool(att_m) # torch.Size([8, 900, 1])
            map = pool_m.view(b, -1, h, w) # torch.Size([8, 1, 30, 30])
            map = torch.sigmoid(map)
            weights = torch.matmul(map, v)
            fea = feature * weights
            out_list.append(fea)

        return out_list
