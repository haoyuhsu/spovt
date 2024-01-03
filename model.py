import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
# from transformers import BertModel, BertTokenizer
from encoder import TransformerEncoder, RelEncoder, PointEncoder, TVAEPointEncoder
from decoder import TransformerDecoder, PointDecoder, PointDecoder2, PointDecoder3, PointDecoder4, TVAEPointDecoder, PointDecoderSimple, TVAEPointRefineDecoder
from embedding import Sentence_Embeddings, Concat_Embeddings, Add_Embeddings


class TVAEPart2Complete(nn.Module):
    def __init__(self, point_dim = 3, max_token=6, noise_size=64,\
                 hidden_size=512, num_layers=5, attn_heads=4, dropout=0.3, pretrain=False, mode = 'pretrain1', out_points = 256):
        super(TVAEPart2Complete, self).__init__()
        self.pretrain = pretrain
        self.encoder = TVAEPointEncoder(point_dim=point_dim, max_token=max_token, hidden_size=hidden_size, num_layers=num_layers, attn_heads=attn_heads, dropout=dropout)
        self.decoder = TVAEPointDecoder(hidden_size=hidden_size,  dropout=0.0, max_token=max_token,num_layers=7)
        self.max_token = max_token
        self.out_points = out_points

    def forward(self, input_points, token_type, label ,src_mask = None, gt_seg = None ,pretrain=False, mode = 'pretrain1', inference = False, self_recon = False ,input_ratio=None, gt_ratio = None, n=None):
        # [B, 4+2048, D]
        bs = input_points.size(0)
        encoder_output, label_logits, ratio_logits, mu, log_std = self.encoder(input_points, token_type, label ,src_mask, pretrain=pretrain, input_ratio = input_ratio)
        if inference:
            if self_recon==True:
                z_bar = self.reparameterize(mu[:,(self.max_token-2):, :], log_std[:,(self.max_token-2):, :], inference)
                out_label = label[:,(self.max_token-2):]
                z = self.denormalize(out_label, z_bar, encoder_output[:,:(self.max_token-2),:])
                input_z = z
                points = self.decoder(z, out_label, input_ratio, z, src_mask)
            else:
                input_z_bar = self.reparameterize(mu[:,(self.max_token-2):, :], log_std[:,(self.max_token-2):, :], inference)
                input_label  = label[:,(self.max_token-2):]
                input_z = self.denormalize(input_label, input_z_bar, encoder_output[:,:(self.max_token-2),:])
                ratio_mask = (ratio_logits >= 0.07).float()
                ratio_logits = ratio_logits.view(bs, -1)
                ratio_logits = ratio_logits*ratio_mask
                ratio_logits /= torch.sum(ratio_logits, dim=1).unsqueeze(1).repeat(1,self.max_token-2)
                out_label = torch.multinomial(ratio_logits, self.out_points, replacement=True)
                z_bar = torch.randn((mu.size(0), self.out_points, mu.size(2))).to(mu.device)
                z = self.denormalize(out_label, z_bar, encoder_output[:,:(self.max_token-2),:])
                points = self.decoder(z, out_label, input_ratio, z, src_mask)
        else:
            if self_recon==True:
                z_bar = self.reparameterize(mu[:,(self.max_token-2):, :], log_std[:,(self.max_token-2):, :])
                out_label = label[:,(self.max_token-2):]
                z = self.denormalize(out_label, z_bar, encoder_output[:,:(self.max_token-2),:])
                input_z = z
                points = self.decoder(z, out_label, input_ratio, z, src_mask)
            else:
                input_z_bar = self.reparameterize(mu[:,(self.max_token-2):, :], log_std[:,(self.max_token-2):, :])
                input_label  = label[:,(self.max_token-2):]
                input_z = self.denormalize(input_label, input_z_bar, encoder_output[:,:(self.max_token-2),:])
                out_label = torch.multinomial(gt_ratio, self.out_points, replacement=True)
                z_bar = torch.randn((mu.size(0), self.out_points, mu.size(2))).to(mu.device)
                z = self.denormalize(out_label, z_bar, encoder_output[:,:(self.max_token-2),:])
                src_mask = torch.ones(bs, 1, self.out_points+self.max_token-2).bool().to(z.device)
                points = self.decoder(z, out_label, input_ratio, z, src_mask)

        return points, out_label, ratio_logits, mu, log_std, input_z

    def reparameterize(self, mu: Tensor, log_std: Tensor, inference = False) -> Tensor:
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        if inference:
            return mu
        else:
            mu = (eps * std) + mu
            return mu
        
    def denormalize(self, labels, samples, prototypes):
        denormalized_samples = torch.zeros_like(samples)
        for i in range(self.max_token-2):
            d_samples = (samples*(torch.abs(prototypes[:,i]).reshape(-1,1,samples.size(-1)).repeat(1,samples.size(1),1)) + prototypes[:,i].unsqueeze(1).repeat(1,samples.size(1),1))*(labels==i).int().to(samples.device).unsqueeze(-1).repeat(1,1,samples.size(-1))
            denormalized_samples += d_samples
        return denormalized_samples


class TVAERefine(nn.Module):
    def __init__(self, point_dim = 3, max_token=6, noise_size=64,\
                 hidden_size=512, num_layers=5, attn_heads=4, dropout=0.3, pretrain=True, mode = 'pretrain1', coarse_points = 256, fine_points = 256, expand = 1):
        super(TVAERefine, self).__init__()
        self.coarse_model = TVAEPart2Complete(max_token=max_token,out_points=coarse_points)
        self.pretrain = pretrain
        self.first_fc = nn.Linear(hidden_size+3, hidden_size)
        self.first_fc_add_reliable = nn.Linear(hidden_size+5, hidden_size)
        self.fine_decoder = TVAEPointRefineDecoder(hidden_size=hidden_size,  dropout=0.0, max_token=max_token,num_layers=4, point_num = fine_points ,expand = expand)
        self.max_token = max_token
        self.coarse_points = coarse_points
        self.fine_points = fine_points
        self.expand = expand

    def forward(self, input_points, token_type, label ,src_mask = None, gt_seg = None ,pretrain=True, mode = 'pretrain1', inference = True, self_recon = False ,input_ratio=None, gt_ratio = None, n=None, add_input=True):
        # [B, 4+2048, D]
        bs = input_points.size(0)
        points, out_label, ratio_logits, mu, log_std, encoder_output = self.coarse_model(input_points, token_type, label , src_mask, gt_seg ,
                                                                        pretrain = pretrain, mode = mode, inference = inference,self_recon=self_recon ,input_ratio = input_ratio, gt_ratio = gt_ratio, n = n)
        input_num = input_points.size(1)-self.max_token+2
        ### add input points
        if add_input:
            point_pad_mask = torch.cat((src_mask.reshape(bs,-1)[:,self.max_token-2:][:,:self.fine_points].long(), torch.zeros_like(out_label)),dim=-1)
            reliable_token = F.one_hot(point_pad_mask.long(), num_classes=2)
            points = torch.cat((input_points[:,self.max_token-2:,:][:,:self.fine_points,:], points), dim=1)
            out_label = torch.cat((label[:,self.max_token-2:][:,:self.fine_points], out_label), dim = 1)
            id_0 = torch.arange(bs).view(-1, 1)
            concat_feat  = self.first_fc_add_reliable(torch.cat((points, encoder_output[id_0,out_label], reliable_token), dim = -1))
        else:
            point_pad_mask = torch.zeros_like(out_label)
            reliable_token = F.one_hot(point_pad_mask.long(), num_classes=2)
            id_0 = torch.arange(bs).view(-1, 1)
            concat_feat  = self.first_fc_add_reliable(torch.cat((points, encoder_output[id_0,out_label], reliable_token), dim = -1))
        src_mask = torch.ones(bs, 1, concat_feat.size(1)+self.max_token-2).bool().to(concat_feat.device)
        displacement = self.fine_decoder(concat_feat, out_label, input_ratio, concat_feat, mask=src_mask).reshape(bs, points.size(1),self.expand, 3)
        if mode == "train":
            if add_input:
                points = points.unsqueeze(-2).repeat(1,1,self.expand,1)#[:,input_num:,:,:]
                out_label = out_label.unsqueeze(-1).repeat(1,1,self.expand).reshape(bs, -1)#[:,self.expand*input_num:]
                return points, displacement ,out_label, ratio_logits, mu, log_std, encoder_output
            else:
                points = points.unsqueeze(-2).repeat(1,1,self.expand,1)
                out_label = out_label.unsqueeze(-1).repeat(1,1,self.expand).reshape(bs, -1)
                return points, displacement ,out_label, ratio_logits, mu, log_std, encoder_output
        elif mode == "test":
            if add_input:
                points = (points.unsqueeze(-2).repeat(1,1,self.expand,1) + displacement).reshape(bs,-1,3)#[:,self.expand*input_num:,:]
                out_label = out_label.unsqueeze(-1).repeat(1,1,self.expand).reshape(bs, -1)#[:,self.expand*input_num:]
            else:
                points = (points.unsqueeze(-2).repeat(1,1,self.expand,1) + displacement).reshape(bs,-1,3)
                out_label = out_label.unsqueeze(-1).repeat(1,1,self.expand).reshape(bs, -1)
            return points ,out_label, ratio_logits, mu, log_std, encoder_output
    
    


        





        
