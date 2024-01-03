import sys
sys.path.append('../')
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TVAEPart2Complete, TVAERefine
from dataset import ShapeNet_Completion_Seg
from manager import Manager


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', help= "Cuda index", default= "0")
    parser.add_argument('-info', default= '')
    parser.add_argument('-epoch', type= int, default= 1000)
    parser.add_argument('-lr', type= float, default= 1e-4)
    parser.add_argument('-bs', help= 'Batch size', type= int, default= 1)
    parser.add_argument('-dataset', default= './data')
    parser.add_argument('-size_limit', type= int, default= None)
    parser.add_argument('-load', default= None)
    parser.add_argument('-save', default= None)
    parser.add_argument('-record', help= 'Record file name', default= 'record.txt')
    parser.add_argument('-interval', type= int, help= 'Record interval within an epoch', default= 100)
    parser.add_argument('-cat', type= str, default='chair')
    parser.add_argument('-mode', type= str, default='test')
    parser.add_argument('-output', type= str, default='car_pretrain')
    parser.add_argument('-point_num', type= int, default=256)
    args = parser.parse_args()
    
    print('== Training ==')
    torch.cuda.set_device(int(args.cuda))
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    train_data = ShapeNet_Completion_Seg(args.dataset, 'train', point_num=args.point_num ,category=args.cat)
    train_loader = DataLoader(train_data, collate_fn=collate_fn, shuffle= True, batch_size= args.bs)
    test_data = ShapeNet_Completion_Seg(args.dataset, 'test',point_num=args.point_num ,category=args.cat)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, shuffle= False, batch_size= int(args.bs))

    max_token = test_data.part_nums + 2
    print('max_token:', max_token)

    if (args.mode == 'pretrain') or (args.mode == 'train'):
        model = TVAEPart2Complete( max_token = max_token)
    else:
        model = TVAERefine(max_token = max_token, fine_points=args.point_num)

    manager = Manager(model, device, args)
    
    if (args.mode == 'pretrain') or (args.mode == 'train'):
        manager.train(train_loader, test_loader)
    elif args.mode == 'test':
        manager.multi_validate(test_loader, refine=True, save=args.output)
    elif args.mode == 'refine':
        manager.train_refine(train_loader, test_loader, add_input=False)
    elif args.mode == 'refine2':
        manager.train_refine(train_loader, test_loader, add_input=True)
    

if __name__ == '__main__':
    main()