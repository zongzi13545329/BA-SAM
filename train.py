# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function 


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

'''load pretrained model'''
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    
    net.load_state_dict(checkpoint['state_dict'],strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])
    print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

args.path_helper = set_log_dir('logs_cvpr_specific', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])


if args.dataset == 'isic':
    '''isic data'''
    isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

elif args.dataset == 'decathlon':
    nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)

elif args.dataset == 'COD10K':
    '''isic data'''
    cod10k_train_dataset = CamObjDataset(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Training')
    cod10k_test_dataset = CamObjDataset(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Test')

    nice_train_loader = DataLoader(cod10k_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(cod10k_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(cod10k_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


elif args.dataset == 'SUB':
    '''isic data'''
    sub_train_dataset = SUB(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    sub_val_dataset = SUB(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'val')
    sub_test_dataset = SUB(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(sub_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(sub_val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(sub_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


elif args.dataset == 'DIS':
    '''isic data'''
    dis_train_dataset = DIS(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Training')
    dis_val_dataset = DIS(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Val')
    dis_test_dataset = DIS(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Test')

    nice_train_loader = DataLoader(dis_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(dis_val_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(dis_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

elif args.dataset == 'DUTS':
    '''isic data'''
    duts_train_dataset = DUTS(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Training')
    duts_test_dataset = DUTS(image_size=args.image_size, out_size = args.out_size, data_path=args.data_path, mode = 'Test')

    nice_train_loader = DataLoader(duts_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_val_loader = DataLoader(duts_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(duts_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
# input_tensor = torch.Tensor(args.b, 3, 256, 256).cuda(device = GPUdevice)
# writer.add_graph(net, Variable(input_tensor, requires_grad=True))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
# best_tol = 1e4
best_mae = 1e4
for epoch in range(settings.EPOCH):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
        # if epoch and epoch % args.save_freq == 0 or epoch == settings.EPOCH-1:
        #     is_best = False
        #     if args.distributed != 'none':
        #         sd = net.module.state_dict()
        #     else:
        #         sd = net.state_dict()
        #     save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': args.net,
        #     'state_dict': sd,
        #     'optimizer': optimizer.state_dict(),
        #     'best_tol': best_tol,
        #     'path_helper': args.path_helper,}, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice, emae) = function.validation_sam(args, nice_val_loader, epoch, net, writer)
            logger.info(f'Eval on DUTS: Total score: {tol}, IOU: {eiou}, DICE: {edice}, MAE: {emae} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            # if tol < best_tol:
            #     best_tol = tol
            #     is_best = True
            if emae < best_mae:
                best_mae = emae
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                # 'best_tol': best_tol,
                'best_mae': best_mae,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint_duts.pth")
            else:
                is_best = False

        logger.info(f'Best MAE on DUTS: {best_mae}')

writer.close()
