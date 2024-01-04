""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

class CamObjDataset(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        if mode == 'Training':
            image_root = os.path.join(self.data_path + '/TrainDataset/Imgs/')
            gt_root = os.path.join(self.data_path + '/TrainDataset/GT/')
        else :
            image_root = os.path.join(self.data_path + '/TestDataset/COD10K/Imgs/')
            gt_root = os.path.join(self.data_path + '/TestDataset/COD10K/GT/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size



class SUB(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        if mode == 'Training':
            df = pd.read_csv(os.path.join(data_path + '/train.txt'), sep=' ', header=None, names=['img', 'mask'])
        elif mode == 'val':
            df = pd.read_csv(os.path.join(data_path + '/val.txt'), sep=' ', header=None, names=['img', 'mask'])
        else:
            df = pd.read_csv(os.path.join(data_path + '/test.txt'), sep=' ', header=None, names=['img', 'mask'])
        
        self.name_list = df['img'].tolist()
        self.label_list = df['mask'].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }



class DIS(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        if mode == 'Training':
            image_root = os.path.join(self.data_path + '/DIS-TR/im/')
            gt_root = os.path.join(self.data_path + '/DIS-TR/gt/')
        elif mode == 'Val':
            image_root = os.path.join(self.data_path + '/DIS-VD/im/')
            gt_root = os.path.join(self.data_path + '/DIS-VD/gt/')       
        else :
            image_root = os.path.join(self.data_path + '/DIS-TE4/im/')
            gt_root = os.path.join(self.data_path + '/DIS-TE4/gt/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class DUTS(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        if mode == 'Training':
            image_root = os.path.join(self.data_path + '/DUTS-TR/DUTS-TR-Image/')
            gt_root = os.path.join(self.data_path + '/DUTS-TR/DUTS-TR-Mask/')
        else :
            image_root = os.path.join(self.data_path + '/DUTS-TE/DUTS-TE-Image/')
            gt_root = os.path.join(self.data_path + '/DUTS-TE/DUTS-TE-Mask/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size



class CDS2K(Dataset):
    #####only have test dataset
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        image_root = os.path.join(self.data_path + '/Positive/Image/')
        gt_root = os.path.join(self.data_path + '/Positive/GroundTruth/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            try:
                pt = random_click(np.array(mask) / 255, point_label, inout)
            except:
                print (self.images[index].split('/')[-1].split(".jpg")[0])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class CAMO(Dataset):
    #####only have test dataset
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        image_root = os.path.join(self.data_path + '/TestDataset/CAMO/Imgs/')
        gt_root = os.path.join(self.data_path + '/TestDataset/CAMO/GT/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class CHAMELEON(Dataset):
    #####only have test dataset
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click', plane = False):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        image_root = os.path.join(self.data_path + '/TestDataset/CHAMELEON/Imgs/')
        gt_root = os.path.join(self.data_path + '/TestDataset/CHAMELEON/GT/')           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])
    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)
        
        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size
        