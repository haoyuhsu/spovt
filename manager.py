import sys
sys.path.append('../')
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_warmup as warmup
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import chamfer_distance as CD
from utils import *
import time
import os
import h5py
import random
from tqdm import tqdm


def get_miou(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list, mask = None):
    valid_labels = valid_labels.tolist()
    if mask == None:
        mask = torch.ones(pred.size())
    pred, target, mask = pred.cpu().numpy(), target.cpu().numpy(), mask.cpu().numpy()
    part_ious = []
    for part_id in valid_labels:
        pred_part = (pred == part_id)*mask
        target_part = (target == part_id)*mask
        I = np.sum(np.logical_and(pred_part, target_part))
        U = np.sum(np.logical_or( pred_part, target_part))
        if U == 0:
            part_ious.append(1)
        else:
            part_ious.append(I/U)
    miou = np.mean(part_ious)
    return miou


def multi_kld(mu, log_std, seg, max_part):
    ### fix unbalanced label
    kld = 0
    for i in range(max_part):
        mask_num  = (seg==i).int().to(mu.device).sum(1)
        mask_mu = mu*(seg == i).int().to(mu.device).unsqueeze(-1).repeat(1,1,mu.size(-1))
        mask_log_std = log_std*(seg == i).int().to(log_std.device).unsqueeze(-1).repeat(1,1,log_std.size(-1))
        kld_i = -0.5 * torch.sum(1 + mask_log_std*2 - mask_mu ** 2 - (mask_log_std*2).exp(), dim = -1).reshape(-1)
        kld += torch.sum(kld_i)/(torch.sum(mask_num)+1e-8)
    return kld


class Manager():
    def __init__(self, model, device, args):
        self.args = args
        self.device = device
        self.info = args.info
        self.epoch = args.epoch
        self.load = args.load
        self.model = model
        self.mode = args.mode
        print("========== {} ==========".format(self.mode))
        if self.mode == 'pretrain':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr= args.lr, weight_decay=0.001)
        else:
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr= args.lr, weight_decay=0.0001)
        self.load = args.load
        if args.load:
            if self.mode == 'refine':
                self.model.coarse_model.load_state_dict(torch.load(args.load, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(args.load, map_location=self.device))
        if self.mode == 'refine':
            for p in self.model.coarse_model.parameters():
                p.requires_grad = False
        elif self.mode =='refine2':
            for p in self.model.coarse_model.encoder.parameters():
                p.requires_grad = False
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= 20, gamma= 0.9)
        self.size_limit = args.size_limit
        self.save = args.save
        self.record_interval = args.interval
        self.record_file_name = args.record
        self.record_file = None
        self.best = { "epoch": 0, "loss": 1000}
        self.cat = args.cat
        self.model = self.model.to(self.device)

    def record(self, info):
        print(info)
        self.record_file.write(info + '\n')
        
    def train(self, train_data, test_data, double=False):
        self.record_file = open(self.record_file_name, 'w')
        self.record("Info : {}".format(self.info))
        self.record("Model parameter number: {}".format(parameter_number(self.model)))
        self.record("Model structure:\n{}".format(self.model.__str__()))
        self.record("---------------- Training records ----------------")
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= 20, gamma= 0.9)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        stop_high = 0.0638
        step = stop_high/40
        v = step
        for epoch in range(self.epoch):
            self.epoch = epoch
            self.model.train()
            train_loss_x1 = 0
            train_loss_y1 = 0
            train_point_kld_loss = 0
            train_sem_kld_loss = 0
            train_ratio_loss = 0
            seg_loss = 0
            train_mse_loss = 0
            miou = 0
            max_token = self.model.max_token
            max_part = max_token-2
            if v<=stop_high:
                beta = (0.5-0.5*np.cos(v*np.pi))*0.05
                v+=step
            else:
                beta = 0
                v = step
            print('beta==',beta)
            for i, (partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio) in tqdm(enumerate(train_data), total=len(train_data), desc="Epoch {}".format(epoch+1)):
                # n : actual partial sequence, points after n are padding points
                partial_point = partial_point.to(self.device)
                partial_seg = partial_seg.long().to(self.device)
                partial_token = partial_token.long().to(self.device)
                gt_point = gt_point.to(self.device)
                gt_seg = gt_seg.long().to(self.device)
                gt_token = gt_token.long().to(self.device)
                gt_part_ratio = gt_part_ratio.to(self.device)
                partial_part_ratio = partial_part_ratio.to(self.device)
                n = n.to(self.device)
                label = label.long().to(self.device)
                bs, v_partial, _ = partial_point.size()
                _, v_gt, _ = gt_point.size()
                partial_src_mask = (partial_token != 0).unsqueeze(1).to(self.device)

                ##### train with self_reconstruction #####
                point, seg, ratio_pred, mu, log_std, _ = self.model(partial_point, partial_token, partial_seg ,partial_src_mask, gt_seg ,pretrain = False, mode = 'train',self_recon=True ,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n)
                
                mse_mask = (seg != -2).unsqueeze(-1).to(self.device).repeat(1,1,3)
                mse_loss = torch.mean(((point*mse_mask)-(partial_point[:,max_part:,:]*mse_mask))**2)
                   
                kld_loss_point = multi_kld(mu[:,max_part:,:], log_std[:,max_part:,:], partial_seg[:,max_part:], max_part)
                kld_loss = kld_loss_point 

                loss2 = ((ratio_pred - gt_part_ratio)**2).mean()

                if self.mode == 'pretrain':
                    loss = (mse_loss*1 + kld_loss*beta )*1.0 + loss2*0.005
                else:
                    loss = (mse_loss*1 + kld_loss*1e-5 )*1 + loss2*0.005
                    loss = loss/1.0
                    loss.backward()
                    train_mse_loss += mse_loss.item()*1.0
                    train_point_kld_loss += kld_loss_point.item()*1.0
                    train_ratio_loss += loss2.item()*1.0

                ##### train with completion #####
                if self.mode=='train':
                    point, seg, ratio_pred, mu, log_std, _ = self.model(partial_point, partial_token, partial_seg ,partial_src_mask, gt_seg ,pretrain = False, mode = 'train',self_recon=False ,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n)
                    cham_x1, cham_y1, _ = chamfer_distance(point.float(), gt_point[:,max_part:,:])
                    id_0 = torch.arange(bs).view(-1,1)
                    gt_near = knn_points(point.float(), gt_point[:, max_part:,:], K=1).idx.squeeze()
                    gt_label = gt_seg[:,max_part:][id_0, gt_near]
                    loss = cham_x1 + cham_y1*1.0
                    loss = loss/1.0
                    loss.backward()

                self.lr_scheduler.step(epoch-1)
                self.warmup_scheduler.dampen()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                seg_pred = seg
                if self.mode == 'pretrain':
                    gt_label = partial_seg[:,max_part:]
                iou = get_miou(seg_pred, gt_label, torch.arange(max_part))

                if self.mode== 'train' :
                    train_loss_x1 += cham_x1.item()
                    train_loss_y1 += cham_y1.item()
                
                seg_loss += 0
                miou += iou
                
                if (i + 1) % self.record_interval == 0:
                    self.record(' epoch {} step {} | average cham: {}/{} | average self recon mse: {}| average ratio loss: {} | average seg: {} | average kld:{}/{} | iou:{} '.format(epoch +1, i+1, train_loss_x1 / (i + 1),train_loss_y1 / (i + 1),train_mse_loss / (i+1),train_ratio_loss/(i+1) ,seg_loss/(i+1), train_point_kld_loss/(i+1),train_sem_kld_loss/(i+1) ,miou/(i+1)))

            train_loss_x1, train_loss_y1 = train_loss_x1/(i+1), train_loss_y1/(i+1)
            train_ratio_loss /= (i+1)
            seg_loss = seg_loss/ (i+1)
            train_point_kld_loss = train_point_kld_loss/(i+1)
            train_sem_kld_loss = train_sem_kld_loss/(i+1)
            train_mse_loss = train_mse_loss/(i+1)
            miou = miou/(i+1)

            torch.cuda.empty_cache()
            del point, seg, ratio_pred, mu, log_std, partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio, iou
            test_loss, test_miou, test_mse = self.validate(test_data)
            self.record('= Epoch {} | Train chamfer Loss: {:.5f}/{:.5f} | Train mse Loss: {:.5f}| Train ratio Loss: {:.5f}| Train KLD: {:.5f}/{:.5f} | Test Loss: {:.5f}/{:.5f}/{:.5f}/{:.5f} | Miou: {:.5f}'.format(epoch + 1, train_loss_x1, train_loss_y1,train_mse_loss , train_ratio_loss, train_point_kld_loss, train_sem_kld_loss ,test_loss[0], test_loss[1], test_loss[2], test_mse,test_miou))

            if self.mode=='train':
                if sum(test_loss[:2]) < self.best['loss']:
                    self.best['epoch'] = epoch + 1
                    self.best['loss'] = sum(test_loss[:2])
                    if self.save:
                        torch.save(self.model.state_dict(), self.save)
            elif self.mode == 'pretrain':
                if (test_mse < self.best['loss']) and (epoch>=0):
                    self.best['epoch'] = epoch + 1
                    self.best['loss'] = test_mse
                    if self.save:
                        torch.save(self.model.state_dict(), self.save)

            print('current best: {}'.format(self.best['loss']))
            torch.save(self.model.state_dict(),self.save.split('.')[0]+'_last.pth')
        self.record('* Best result at {} epoch with test acc {}'.format(self.best['epoch'], self.best['loss']))

    def validate(self, test_data, refine = False, add_input=True):
        self.model.eval()
        max_token = self.model.max_token
        max_part = max_token-2
        test_loss_x = 0
        test_loss_y = 0
        test_ratio_loss = 0
        test_mse_loss = 0
        miou = 0
        cnt = 0
        with torch.no_grad():
            for i, (partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio) in enumerate(test_data):
                # n : actual partial sequence, points after n are padding points
                partial_point = partial_point.to(self.device).detach()
                partial_seg = partial_seg.long().to(self.device).detach()
                partial_token = partial_token.long().to(self.device).detach()
                gt_point = gt_point.to(self.device).detach()
                gt_seg = gt_seg.long().to(self.device).detach()
                gt_token = gt_token.long().to(self.device).detach()
                label = label.long().to(self.device).detach()
                gt_part_ratio = gt_part_ratio.to(self.device).detach()
                partial_part_ratio = partial_part_ratio.to(self.device).detach()
                n = n.to(self.device).detach()
                bs, v_partial, _ = partial_point.size()
                _, v_gt, _ = gt_point.size()
                partial_src_mask = (partial_token != 0).unsqueeze(1).to(self.device).detach()
                cnt += bs

                if refine==False:
                    ##### self recon part #####
                    point, seg, ratio_pred, mu, _, _ = self.model(partial_point, partial_token, partial_seg, partial_src_mask, gt_seg, pretrain = False, mode = 'test',inference=False, self_recon=True,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n)
                    point, seg, ratio_pred = point.detach(), seg.detach(), ratio_pred.detach()
                    mse_mask = (seg != -2).unsqueeze(-1).to(self.device).repeat(1,1,3)
                    mse_loss = torch.mean(((point*mse_mask)-(partial_point[:,max_part:,:]*mse_mask))**2).cpu()
                    test_mse_loss += mse_loss.item()*bs
                    del point, seg, ratio_pred, mse_loss, mse_mask, mu
                    torch.cuda.empty_cache()
                    ##### completion part #####
                    point, seg, ratio_pred, _, _, _ = self.model(partial_point, partial_token, partial_seg, partial_src_mask, gt_seg, pretrain = False, mode = 'test',inference=False, self_recon=False,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n)
                else:
                    point, seg, ratio_pred, _, _, _ = self.model(partial_point, partial_token, partial_seg, partial_src_mask, gt_seg, pretrain = False, mode = 'test',inference=False, self_recon=False,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n, add_input=add_input)
                
                point, seg, ratio_pred = point.detach(), seg.detach(), ratio_pred.detach()
                cham_x, cham_y, _ = chamfer_distance(point.float(), gt_point[:,max_part:,:], sqrt=False)
                loss2 = ((ratio_pred - gt_part_ratio)**2).mean()
                torch.cuda.empty_cache()
                id_0 = torch.arange(bs).view(-1,1)
                gt_near = knn_points(point.float(), gt_point[:, max_part:,:], K=1).idx.squeeze()
                gt_label = gt_seg[:,max_part:][id_0, gt_near]
                seg_pred = seg
                iou = get_miou(seg_pred, gt_label, torch.arange(max_part))     
                test_loss_x += cham_x.item()*bs
                test_loss_y += cham_y.item()*bs
                test_ratio_loss += loss2.item()*bs
                miou += iou.item()*bs
                torch.cuda.empty_cache()

        test_loss_x /= cnt
        test_loss_y /= cnt
        test_ratio_loss /= cnt
        test_mse_loss /= cnt
        miou = miou/cnt
        # print(test_loss_x, test_loss_y, test_ratio_loss, test_mse_loss, miou)
        
        return (test_loss_x, test_loss_y, test_ratio_loss), miou, test_mse_loss
    
    def train_refine(self, train_data, test_data, double=False, add_input=True):
        self.record_file = open(self.record_file_name, 'w')
        self.record("Info : {}".format(self.info))
        self.record("Model parameter number: {}".format(parameter_number(self.model)))
        self.record("Model structure:\n{}".format(self.model.__str__()))
        self.record("---------------- Training records ----------------")
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size= 20, gamma= 0.9)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        torch.autograd.set_detect_anomaly(True)
   
        for epoch in range(self.epoch):
            self.epoch = epoch
            self.model.train()
            train_loss_x1 = 0
            train_loss_y1 = 0
            train_kld_loss = 0
            train_ratio_loss = 0
            seg_loss = 0
            train_mse_loss = 0
            miou = 0
            max_token = self.model.max_token
            max_part = max_token-2
            train_loss_x2 = 0
            train_loss_y2 = 0
            fidality = 0
            for i, (partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio) in tqdm(enumerate(train_data), total=len(train_data), desc="Epoch {}".format(epoch+1)):
                # n : actual partial sequence, points after n are padding points
                partial_point = partial_point.to(self.device).detach()
                partial_seg = partial_seg.long().to(self.device)
                partial_token = partial_token.long().to(self.device)
                gt_point = gt_point.to(self.device)
                gt_seg = gt_seg.long().to(self.device)
                gt_token = gt_token.long().to(self.device)
                gt_part_ratio = gt_part_ratio.to(self.device)
                partial_part_ratio = partial_part_ratio.to(self.device)
                n = n.to(self.device)
                label = label.long().to(self.device)
                bs, v_partial, _ = partial_point.size()
                _, v_gt, _ = gt_point.size()
                partial_src_mask = (partial_token != 0).unsqueeze(1).to(self.device)                

                ##### train with completion #####
                point, displacement ,seg, ratio_pred, mu, log_std, _ = self.model(partial_point, partial_token, partial_seg ,partial_src_mask, gt_seg ,pretrain = False, mode = 'train',self_recon=False ,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n, add_input=add_input)
                point_fine = (point + displacement).reshape(bs,-1,3)
                if self.mode=='refine2':
                    cham_x2, cham_y2, _ = chamfer_distance(point[:,:].reshape(bs,-1,3).float(), gt_point[:,max_token-2:,:])
                cham_x1, cham_y1, _ = chamfer_distance(point_fine.float()[:,:], gt_point[:,max_token-2:,:])
                id_0 = torch.arange(bs).view(-1,1)
                gt_near = knn_points(point_fine.float(), gt_point[:, max_token-2:,:], K=1).idx.squeeze()
                gt_label = gt_seg[:, max_token-2:][id_0, gt_near]
               
                if self.mode=='refine2':
                    loss = (cham_x1*1.0 + cham_y1 + cham_x2*1.0 + cham_y2*1 )*1000 #+ f_loss*2
                elif self.mode=='refine':
                    loss = (cham_x1*1.0 + cham_y1)*1000

                loss = loss/1.0
                loss.backward()

                self.lr_scheduler.step(epoch-1)
                self.warmup_scheduler.dampen()

                self.optimizer.step()
                self.optimizer.zero_grad()
                seg_pred = seg
                iou = get_miou(seg_pred, gt_label, torch.arange(max_token-2))

                train_loss_x1 += cham_x1.item()
                train_loss_y1 += cham_y1.item()
                if self.mode == 'refine2':
                    train_loss_x2 += cham_x2.item()
                    train_loss_y2 += cham_y2.item()
                
                seg_loss += 0
                miou += iou
                
                if (i + 1) % self.record_interval == 0:
                    self.record(' epoch {} step {} | average cham: {}/{} | average self recon mse: {}| average ratio loss: {} | average seg: {} | average kld:{} | iou:{} '.format(epoch +1, i+1, train_loss_x1 / (i + 1),train_loss_y1 / (i + 1),train_mse_loss / (i+1),train_ratio_loss/(i+1) ,seg_loss/(i+1), train_kld_loss/(i+1) ,miou/(i+1)))

            train_loss_x1, train_loss_y1 = train_loss_x1/(i+1), train_loss_y1/(i+1)
            train_loss_x2, train_loss_y2 = train_loss_x2/(i+1), train_loss_y2/(i+1)
            train_ratio_loss /= (i+1)
            seg_loss = seg_loss/ (i+1)
            train_kld_loss = train_kld_loss/(i+1)
            train_mse_loss = train_mse_loss/(i+1)
            miou = miou/(i+1)
            fidality /= (i+1)
            torch.cuda.empty_cache()
            del point, point_fine,seg, ratio_pred, mu, log_std, partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio, iou
            test_loss, test_miou, test_mse = self.validate(test_data, refine=True, add_input=add_input)
            self.record('= Epoch {} | Train chamfer Loss: {:.5f}/{:.5f} | Train mse Loss: {:.5f}| Train ratio Loss: {:.5f}| Train KLD: {:.5f} | Test Loss: {:.5f}/{:.5f}/{:.5f}/{:.5f} | Miou: {:.5f}'.format(epoch + 1, train_loss_x1, train_loss_y1,train_mse_loss , train_ratio_loss, train_kld_loss ,test_loss[0], test_loss[1], test_loss[2], test_mse,test_miou))

            if sum(test_loss[:2]) < self.best['loss']:
                self.best['epoch'] = epoch + 1
                self.best['loss'] = sum(test_loss[:2])
                if self.save:
                    torch.save(self.model.state_dict(), self.save)
            print('current best: {}'.format(self.best['loss']))
            torch.save(self.model.state_dict(),self.save.split('.')[0]+'_last.pth')

        self.record('* Best result at {} epoch with test acc {}'.format(self.best['epoch'], self.best['loss']))

    def multi_validate(self, test_data, refine = False, save='test'):
        self.model.eval()
        max_token = self.model.max_token
        max_part = max_token-2
        test_loss_x = 0
        test_loss_y = 0
        test_ratio_loss = 0
        test_mse_loss = 0
        miou = 0
        
        all_data_num = test_data.dataset.__len__()
        point = torch.zeros(all_data_num, 16384 , 3).to(self.device)
        seg = torch.zeros(all_data_num,16384).to(self.device)
        gt_all = torch.zeros(all_data_num, 16384, 3).to(self.device)
        gt_seg_all = torch.zeros(all_data_num,16384).to(self.device)
        partial_all = torch.zeros(all_data_num, 512*16,3).to(self.device)
        partial_seg_all = torch.zeros(all_data_num, 512*16).to(self.device)
        ratio_all = torch.zeros(all_data_num,max_part).to(self.device)
        ratio_gt_all = torch.zeros(all_data_num,max_part).to(self.device)

        for j in tqdm(range(16*2), desc='Processing'):
            cnt = 0
            with torch.no_grad():
                for i, (partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio) in enumerate(test_data):
                    # n : actual partial sequence, points after n are padding points
                    partial_point = partial_point.to(self.device).detach()
                    partial_seg = partial_seg.long().to(self.device).detach()
                    partial_token = partial_token.long().to(self.device).detach()
                    gt_point = gt_point.to(self.device).detach()
                    gt_seg = gt_seg.long().to(self.device).detach()
                    gt_token = gt_token.long().to(self.device).detach()
                    label = label.long().to(self.device).detach()
                    gt_part_ratio = gt_part_ratio.to(self.device).detach()
                    partial_part_ratio = partial_part_ratio.to(self.device).detach()
                    n = n.to(self.device).detach()
                    bs, v_partial, _ = partial_point.size()
                    _, v_gt, _ = gt_point.size()
                    partial_src_mask = (partial_token != 0).unsqueeze(1).to(self.device).detach()
                    gt_all[cnt:cnt+bs,:,:] = gt_point[:,max_token-2:,:]
                    gt_seg_all[cnt:cnt+bs,:] = gt_seg[:,max_token-2:]
                    ratio_gt_all[cnt:cnt+bs] += gt_part_ratio
                    partial_all[cnt:cnt+bs, j*256:(j+1)*256,:] = partial_point[:,max_token-2:,:]
                    partial_seg_all[cnt:cnt+bs, j*256:(j+1)*256] = partial_seg[:,max_token-2:]

                    ##### dense output part #####
                    point_fragment, seg_fragment, pred_ratio, _, _, _ = self.model(partial_point, partial_token, partial_seg, partial_src_mask, gt_seg, pretrain = False, mode = 'test',inference=True, self_recon=False,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n)
                    point_fragment, seg_fragment, pred_ratio = point_fragment.detach(), seg_fragment.detach(), pred_ratio.detach()
                    ratio_all[cnt:cnt+bs] += pred_ratio
                    point[cnt:cnt+bs,j*512:(j+1)*512,:] = point_fragment[:,:]
                    seg[cnt:cnt+bs,j*512:(j+1)*512] = seg_fragment[:,:]
                    cnt += bs

        ratio_all /= (j+1)
        ratio_gt_all /= (j+1)
        ratio_loss = ((ratio_gt_all-ratio_all)**2).mean()

        cham_x, cham_y, _ = chamfer_distance(point.float(), gt_all, sqrt=False)
        torch.cuda.empty_cache()
        id_0 = torch.arange(all_data_num).view(-1,1)
        gt_near = knn_points(point.float(), gt_all, K=1).idx.squeeze()
        gt_label = gt_seg_all[id_0, gt_near]
        seg_pred = seg
        iou_pred_to_gt = get_miou(seg_pred, gt_label, torch.arange(max_part))
        id_0 = torch.arange(all_data_num).view(-1,1)
        pred_near = knn_points( gt_all, point.float(),K=1).idx.squeeze()
        seg_label = seg[id_0, pred_near]
        iou_gt_to_pred = get_miou(seg_label, gt_seg_all, torch.arange(max_part))
        iou = (iou_pred_to_gt+iou_gt_to_pred)/2.0

        test_loss_x += cham_x.item()
        test_loss_y += cham_y.item()
        miou += iou.item()
        torch.cuda.empty_cache()

        point = torch.cat((point, seg.view(all_data_num,-1,1)), dim=2)
        partial_all = torch.cat((partial_all, partial_seg_all.view(all_data_num,-1,1)), dim=2)
        gt_all = torch.cat((gt_all, gt_seg_all.view(all_data_num,-1,1)), dim=2)

        for l in tqdm(range(all_data_num), desc='Saving'):
            if not os.path.isdir('visualize/{}/'.format(save)):
                os.makedirs('visualize/{}/'.format(save))
                os.makedirs('seg_image/{}/'.format(save))
            pred_img_path = 'seg_image/{}/pred_{}.png'.format(save, int(l))
            partial_img_path = 'seg_image/{}/partial_{}.png'.format(save, int(l))
            gt_img_path = 'seg_image/{}/gt_{}.png'.format(save, int(l))
            visualize(point.cpu().detach().numpy()[l][:16384,:3], seg.cpu().detach().numpy()[l][:16384], pred_img_path)
            visualize(partial_all.cpu().detach().numpy()[l][:16384], partial_seg_all.cpu().detach().numpy()[l][:16384], partial_img_path)
            visualize(gt_all.cpu().detach().numpy()[l][:16384], gt_seg_all.cpu().detach().numpy()[l][:16384], gt_img_path)
            np.save('visualize/{}/pred_{}.npy'.format(save,int(l)), point.cpu().detach().numpy()[l][:,:])
            np.save('visualize/{}/partial_{}.npy'.format(save,int(l)), partial_all.cpu().detach().numpy()[l][:,:])
            np.save('visualize/{}/gt_{}.npy'.format(save,int(l)), gt_all.cpu().detach().numpy()[l][:,:])

        test_loss_x /= 1
        test_loss_y /= 1
        test_ratio_loss /= 1
        test_mse_loss /= 1
        miou = miou
        return (test_loss_x, test_loss_y, test_ratio_loss), miou, test_mse_loss
    
    def interpolation(self, test_data, refine = False, num_of_samples = 10, gap = 0.2, save=None, part_wise=True):
        self.model.eval()
        max_token = self.model.max_token
        max_part = max_token-2
        test_loss_x = 0
        test_loss_y = 0
        test_ratio_loss = 0
        test_mse_loss = 0
        miou = 0
        if part_wise:
            inter_num = int(1/gap) + 2
        else:
            inter_num = int(1/gap) + 1
        
        all_data_num = test_data.dataset.__len__()
        all_list = torch.arange(all_data_num).tolist()
        
        part_ids = torch.from_numpy(np.random.choice(max_part, num_of_samples)).to(self.device)
        for k in range(num_of_samples):
            pair = random.sample(all_list, 2)
            point = torch.zeros(inter_num, 16384 , 3).to(self.device)
            seg = torch.zeros(inter_num,16384).to(self.device)
            for j in range(32):
                cnt = 0
                with torch.no_grad():
                    for i, (partial_point, partial_seg, partial_token, gt_point, gt_seg, gt_token, label, n, gt_part_ratio, partial_part_ratio) in enumerate(test_data):
                        # n : actual partial sequence, points after n are padding points
                        partial_point = partial_point.to(self.device).detach()
                        partial_seg = partial_seg.long().to(self.device).detach()
                        partial_token = partial_token.long().to(self.device).detach()
                        gt_point = gt_point.to(self.device).detach()
                        gt_seg = gt_seg.long().to(self.device).detach()
                        gt_token = gt_token.long().to(self.device).detach()
                        label = label.long().to(self.device).detach()
                        gt_part_ratio = gt_part_ratio.to(self.device).detach()
                        partial_part_ratio = partial_part_ratio.to(self.device).detach()
                        n = n.to(self.device).detach()
                        bs, v_partial, _ = partial_point.size()
                        _, v_gt, _ = gt_point.size()
                        partial_src_mask = (partial_token != 0).unsqueeze(1).to(self.device).detach()
                        ##### dense output part #####
                        point_fragment, seg_fragment, _, _, _, _ = self.model.interpolation(partial_point, partial_token, partial_seg, partial_src_mask, gt_seg, pretrain = False, mode = 'test',inference=False, self_recon=False,input_ratio = partial_part_ratio, gt_ratio = gt_part_ratio, n = n,  pair=pair, gap=gap, part_wise=part_wise, part_ids=part_ids[k])
                        point_fragment, seg_fragment = point_fragment.detach(), seg_fragment.detach()
                        point[cnt:cnt+bs,j*512:(j+1)*512,:] = point_fragment[:,:]
                        seg[cnt:cnt+bs,j*512:(j+1)*512] = seg_fragment[:,:]
                        cnt += bs
           
            point = torch.cat((point, seg.view(inter_num,-1,1)), dim=2)
            for l in range(inter_num):
                if not os.path.isdir('visualize/{}_interpolation/{}/'.format(save,int(k))):
                    os.makedirs('visualize/{}_interpolation/{}/'.format(save,int(k)))
                    os.makedirs('seg_image/{}_interpolation/{}/'.format(save,int(k)))
                if part_wise:
                    pred_path = 'seg_image/{}_interpolation/{}/{}_{}.png'.format(save,int(k), int(l), part_ids[k])
                else:
                    pred_path = 'seg_image/{}_interpolation/{}/{}.png'.format(save,int(k), int(l))
                visualize(point.cpu().detach().numpy()[l], seg.cpu().detach().numpy()[l], pred_path)
                if part_wise:
                    np.save('visualize/{}_interpolation/{}/pred_{}_{}.npy'.format(save,int(k),int(l), part_ids[k]), point.cpu().detach().numpy()[l][:,:])
                else:
                    np.save('visualize/{}_interpolation/{}/pred_{}.npy'.format(save,int(k),int(l)), point.cpu().detach().numpy()[l][:,:])


##### for visualization #####
import numpy as np
import os
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

COLORS = ["tomato", "forestgreen", "royalblue", "gold", "cyan", "gray"]

def normalize(points: "numpy array (vertice_num, 3)"):
    center = np.mean(points, axis= 0)
    points = points - center
    max_d = np.sqrt(np.max(points @ (points.T)))
    points = points / max_d
    return points 


def visualize(points: "(vertice_num, 3)", labels: "(vertice_num, )", fig_name: str):
    points = np.array(points)[:,:3]
    labels = np.array(labels).astype(int)

    points = normalize(points)
    eye = np.eye(3)
    bound_points = np.vstack((eye , eye * (-1)))
   
    x ,y ,z = points[:, 0], points[:, 1], points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection= "3d")
    ax.axis("off")
    
    colors = [COLORS[i % len(COLORS)] for i in labels]
    ax.scatter(x ,z, y, s= 3, c= colors, marker= "o")
    ax.scatter(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], s=0.01, c= "white")
    plt.savefig(fig_name)
    plt.close()


def weight_visualize(points: "(vertice_num, 3)",query_point_index ,weight: "(vertice_num, )", fig_name: str):
    points = np.array(points)
    labels = np.array(labels)

    points = normalize(points)
    eye = np.eye(3)
    bound_points = np.vstack((eye , eye * (-1)))
   
    x ,y ,z = points[:, 0], points[:, 1], points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection= "3d")
    ax.axis("off")
    
    colors = [COLORS[i % len(COLORS)] for i in labels]
    ax.scatter(x ,z, y, s= 3, c= colors, marker= "o")
    ax.scatter(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], s=0.01, c= "white")
    plt.savefig(fig_name)
    plt.close()


if __name__ == "__main__":
    pass
        
