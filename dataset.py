import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
import random
matplotlib.use('Agg')
from pyntcloud import PyntCloud
import h5py
from pytorch3d.ops.knn import knn_gather, knn_points


class ShapeNet_Completion_Seg(Dataset):
    def __init__(self, dataset_path, mode='train', point_num=256, category='chair', transform=None, shift=False, rotation=False,
    scaling=False, drop=False, mask_input=None , gt_num=16384):
        super().__init__()
        self.transform = transform
        self.shift = shift
        self.rotation = rotation
        self.scaling = scaling
        self.drop = drop
        self.point_num = point_num
        categories_all = {
            "airplane": "02691156",                        
            "car": "02958343",
            "chair": "03001627",                          
            "lamp": "03636649",
            "table": "04379243",
        }
        self.PART_NUM = {
            "airplane": 4,
            "bag": 2,
            "cap": 2,
            "car": 4,
            "chair": 4,
            "earphone": 3,
            "guitar": 3,
            "knife": 2,
            "lamp": 4,
            "laptop": 2,
            "motorbike": 6,
            "mug": 2,
            "pistol": 3,
            "rocket": 3,
            "skateboard": 3,
            "table": 3,
        }
        PART_MIN = {
            "airplane": 0,
            "bag": 4,
            "cap": 6,
            "car": 8,
            "chair": 12,
            "earphone": 16,
            "guitar": 19,
            "knife": 22,
            "lamp": 24,
            "laptop": 28,
            "motorbike": 30,
            "mug": 36,
            "pistol": 38,
            "rocket": 41,
            "skateboard": 44,
            "table": 47,
        }

        if category:
            if category not in categories_all.keys():
                raise Exception('Categoty not found !')
            self.categories = [category]
        else:
            self.categories = categories_all.keys()
        self.categories = list(self.categories)
        print(self.categories)

        self.path_label_pairs = []
        for category in self.categories:
            label = list(categories_all.keys()).index(category)
            min_part = PART_MIN[category]
            folder_path = os.path.join(dataset_path ,mode)
            object_path = [os.path.join(folder_path, categories_all[category] ,'{}'.format(name)) for name in sorted(os.listdir(os.path.join(folder_path,categories_all[category] ))) if name != '.DS_Store']
            for path in object_path[:]:
                partial_list = [f for f in os.listdir(os.path.join(path, 'partial')) if f.endswith('.npy')]
                for part_id in partial_list:
                    pair = (path, os.path.join(path,'partial',part_id), label, min_part)
                    self.path_label_pairs.append(pair)
        print('mode:', mode)
        print('number of data', len(self.path_label_pairs))

        self.part_nums = self.PART_NUM[category]
        self.mask_input = mask_input

    def __len__(self):
        return len(self.path_label_pairs)

    def __getitem__(self, index):
        #############################
        # token 0 = padding
        # token 1 = mask
        # other tokens = original seg_label + 2
        #############################
        obj_path, partial_path ,label, min_part = self.path_label_pairs[index]
        label = torch.LongTensor([label])
        gt_point_seg_full = np.load(obj_path + '/gt_with_seg.npy')
        gt_seg_full = torch.FloatTensor(gt_point_seg_full[:, -1]) - min_part

        gt_index = torch.randperm(16384)
        gt_point_seg = torch.FloatTensor(gt_point_seg_full[gt_index][:16384, :])
        gt_point, gt_seg = gt_point_seg[:,:3], gt_point_seg[:,-1] - min_part
        gt_token = gt_seg + 2

        # count part ratio for gt input
        gt_part_count = torch.unique(gt_seg_full, return_counts=True)
        gt_part_ratio = torch.zeros(self.part_nums)
        for i in range(len(gt_part_count[0])):
            gt_part_ratio[gt_part_count[0].long()[i]] = gt_part_count[1][i]
        gt_part_ratio = gt_part_ratio / 16384

        partial_point = np.load(partial_path)
        partial_point = torch.FloatTensor(partial_point)
        n, _ = partial_point.size()

        if n == 0:
            print(partial_path)
            return None
        if n < self.point_num:
            padding = torch.zeros((self.point_num-n, 4))
            padding[:, -1] -= (2 - min_part)
            partial_point = torch.cat((partial_point, padding))
        elif n > self.point_num:
            sample_index = torch.randperm(n)
            partial_point = partial_point[sample_index]
            partial_point = partial_point[:self.point_num,:]
            n = self.point_num
        partial_point, partial_seg = partial_point[:, :3], partial_point[:, -1] - min_part
        partial_token = partial_seg + 2

        # count part ratio for partial input
        partial_part_count = torch.unique(partial_seg, return_counts=True)
        partial_part_ratio = torch.zeros(self.part_nums)
        
        if -2 in partial_part_count[0]:
            partial_part_count = (partial_part_count[0][1:], partial_part_count[1][1:])
        for i in range(len(partial_part_count[0])):
            partial_part_ratio[partial_part_count[0].long()[i]] = partial_part_count[1][i]
        partial_part_ratio = partial_part_ratio.float() / partial_part_count[1].sum()
       
        if self.mask_input:
            for i in range(len(partial_token)):
                prob = random.random()
                if partial_token[i] != 0 and (prob>self.mask_input):
                    partial_token[i] = 1
        
        special_token = torch.zeros((self.part_nums,3))
        special_token_label = torch.arange(self.part_nums).float()

        #############################
        # Return data and label     #
        # partial_point             #
        # partial_seg               #
        # partial_token             #
        # gt_point                  #
        # gt_seg                    #
        # gt_token                  #
        # label                     #
        # part_nums                 #
        # gt_part_ratio             #
        # partial_part_ratio        #
        #############################
        return torch.cat((special_token,partial_point)), torch.cat((special_token_label,partial_seg)), torch.cat((special_token_label+2,partial_token)), \
        torch.cat((special_token,gt_point)), torch.cat((special_token_label,gt_seg)), torch.cat((special_token_label+2,gt_token)), label, n+self.part_nums, gt_part_ratio, partial_part_ratio


if __name__ == "__main__":
    shapenet_set = ShapeNet_Completion_Seg(dataset_path='../../shapenet_completion_full_with_part', mode='test',category='lamp')