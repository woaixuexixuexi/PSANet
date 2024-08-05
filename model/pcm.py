from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.util import intersectionAndUnionGPU

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()
        self.t = 1
        self.n = 300
        self.m = 250
        self.t_b = 1
        self.n_b = 1200
        self.m_b = 1000
        self.alpha = 0.001

    def contrastive(self, Q_GT_pro, supp_pro_avg, neg_dict, labels_, num_classes, a=0, b=0):
        loss = torch.zeros(1).cuda()

        for q_p, s_p, cls in zip(Q_GT_pro, supp_pro_avg, labels_):  
            q_id = cls.item() + 1
            negative_contrast = torch.tensor([]).cuda()
            positive_dot_contrast = torch.div(F.cosine_similarity(s_p.to('cuda:0'), q_p.to('cuda:0'), 0), self.t)  
            for k in range(1, num_classes): 
                if k in neg_dict.keys() and k != q_id:
                    negative_samples = neg_dict[k]  # torch.Size([4, 256])
                else:
                    continue
                if negative_samples.shape[0] > self.m:  
                    perm = torch.randperm(negative_samples.shape[0])
                    negative_samples = negative_samples[perm[:self.m]]
                negative_dot_contrast = torch.div(F.cosine_similarity(s_p.to('cuda:0'), torch.transpose(negative_samples, 0, 1).to('cuda:0'), 0),
                                                  self.t)  

                temp = negative_contrast
                negative_contrast = torch.cat((temp.to('cuda:0'), negative_dot_contrast.to('cuda:0')), 0)

            pos_logits = torch.exp(positive_dot_contrast)
            neg_logits = torch.exp(negative_contrast).sum()
            mean_log_prob_pos = - torch.log((pos_logits.to('cuda:0') / (neg_logits.to('cuda:0'))) + 1e-8) 

            loss = loss.to('cuda:0') + mean_log_prob_pos.mean().to('cuda:0')

        return loss / len(labels_)

    def contrastive_bg(self, Q_GT_pro, supp_pro_avg, neg_dict, a=0, b=0):
        loss = torch.zeros(1).cuda()
        lens = Q_GT_pro.size(0)

        for q_p, s_p in zip(Q_GT_pro, supp_pro_avg):  
            positive_dot_contrast = torch.div(F.cosine_similarity(s_p.to('cuda:0'), q_p.to('cuda:0'), 0), self.t_b)  
            negative_samples = neg_dict[0] 
            if negative_samples.shape[0] > self.m_b:  
                perm = torch.randperm(negative_samples.shape[0])
                negative_samples = negative_samples[perm[:self.m_b]]
            negative_dot_contrast = torch.div(F.cosine_similarity(s_p.to('cuda:0'), torch.transpose(negative_samples, 0, 1).to('cuda:0'), 0),
                                              self.t_b)  
            mar = self.margin(positive_dot_contrast)

            pos_logits = torch.exp(positive_dot_contrast.to('cuda:0') - torch.tensor(mar).to('cuda:0')) 
            neg_logits = self.focal(negative_dot_contrast) * 0.25
            mean_log_prob_pos = - torch.log((pos_logits.to('cuda:0') / (pos_logits + neg_logits).to('cuda:0')) + 1e-8)  #

            loss = loss.to('cuda:0') + mean_log_prob_pos.mean().to('cuda:0')

        return loss / lens

    def focal(self, sim, a=1.5, b=1.5):
        neg = ((1 / (a+torch.exp(-b*sim))) * torch.exp(sim)).sum()
        return neg

    def margin(self, sim, m=0.3):
        if sim<=0.5:
            mar = m*(1+10*(0.5-sim)**2)
        else:
            mar = m
        return mar

    def _negative_construct(self, pros, c_mask):
        pro_dict = dict()
        for i in range(len(c_mask)):
            if c_mask[i]:
                pro_dict[i] = pros[i].contiguous().view(-1, 256)
            else:
                continue
        return pro_dict

    def forward(self, que_feat, mask_list, y_b, negative_dict):  # feats:4,256,30,30 ; labels:4,473,473 ; predict:4,260,260
        num_classes = mask_list.size(1)
        bs = que_feat.size(0)
        bg_item = 0
        c_id_array = torch.arange(num_classes, device='cuda')

        for bs_ in range(bs):
            proto_list = {}
            c_mask = torch.zeros_like(c_id_array, dtype=torch.bool)

            que_feat_ = que_feat[bs_, :, :, :]  # c x h x w
            que_mask_b = mask_list[bs_, :, :, :]  # c x h x w torch.Size([16, 30, 30])
            label_b = y_b[bs_, :, :]

            que_mask_b = que_mask_b.permute(1, 2, 0).contiguous()
            que_mask_b = que_mask_b.cpu().detach().numpy()
            label = np.argmax(que_mask_b, axis=2) 
            label = torch.from_numpy(label)
            label = label.cuda().unsqueeze(0).unsqueeze(0)  # 1 x 1 x h x w

            for k in range(num_classes):
                label_ = label.clone()
                label_ = torch.where(label_ != k, 0, 1) 
                is_not_all_zeros = torch.all(label_ == 0).item()
                if not is_not_all_zeros:
                    label_b_p = F.interpolate(label_.float(), size=(label_b.size(0), label_b.size(1)), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
                    label_b_ = torch.where(label_b != k, 0, 1)
                    intersection, union, target = intersectionAndUnionGPU(label_b_p, label_b_, 2,255)
                    accuracy = intersection / (target + 1e-10)
                    if accuracy[1] > 0.8:
                        proto_type = Weighted_GAP(que_feat_, label_).squeeze(-1) #torch.Size([1, 256, 1])
                        proto_list[k] = proto_type
                        c_mask[k] = True

            Q_dict = self._negative_construct(proto_list, c_mask)

            if bg_item in Q_dict.keys():
                key = bg_item
                if key not in negative_dict and key == bg_item:
                    negative_dict[key] = Q_dict[key].detach()
                elif key == bg_item:
                    orignal_value = negative_dict[key]
                    negative_dict[key] = torch.cat((Q_dict[key].to('cuda:0'), orignal_value.to('cuda:0')), 0).detach()
                    if key == 0 and negative_dict[key].shape[0] > self.n_b:
                        negative_dict[key] = negative_dict[key][:self.n_b, :]

            for key in Q_dict.keys():
                if key not in negative_dict and key != bg_item:
                    negative_dict[key] = Q_dict[key].detach()
                elif key != bg_item:
                    orignal_value = negative_dict[key]
                    if negative_dict[key].shape[0] > self.m:
                        orignal_value_m = torch.mean(orignal_value, dim=0).unsqueeze(0)
                        new_dict = self.alpha * orignal_value_m.to('cuda:0') + (1 - self.alpha) * Q_dict[key].to('cuda:0')
                        negative_dict[key] = torch.cat((new_dict.to('cuda:0'), orignal_value.to('cuda:0')), 0).detach()
                    else:
                        negative_dict[key] = torch.cat((Q_dict[key].to('cuda:0'), orignal_value.to('cuda:0')), 0).detach()
                    if key != 0 and negative_dict[key].shape[0] > self.n: 
                        negative_dict[key] = negative_dict[key][:self.n, :]

        return negative_dict

if __name__ == '__main__':
    query_feat=torch.randn(2, 256, 30, 30).cuda()
    y_m=torch.randn(2, 473, 473).cuda()
    base_out_soft=torch.randn(2, 16, 30, 30).cuda()
    x = torch.randn(256, 30, 30).cuda()
    supp_list=[x]
    cat_idx=torch.tensor([[ 7, 14]]).cuda()
    contrast_loss=PrototypeContrastLoss()
    negative_dict=dict()
    loss, negative_dict = contrast_loss(query_feat,base_out_soft, supp_list, cat_idx,negative_dict)