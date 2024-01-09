# BA-SAM: Scalable Bias-Mode Attention Mask for Segment Anything Model

The official code for BA-SAM. This method is elaborated in the paper xxx.


## Requirement

``conda env create -f environment.yml``

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

Donwload [MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt), and put it at ./checkpoint/ 

## Getting Started
### Melanoma Segmentation from Skin Images (2D)

1. Download ISIC dataset part 1 from https://challenge.isic-archive.com/data/. Then put the csv files in "./data/ISIC" under your data path. Your dataset folder under "your_data_path" should be like:

ISIC/

     ISBI2016_ISIC_Part1_Test_Data/...
     
     ISBI2016_ISIC_Part1_Training_Data/...
     
     ISBI2016_ISIC_Part1_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part1_Training_GroundTruth.csv
   
2. Begin Adapting! run: ``python train.py -net sam -mod sam_adpt -exp_name *msa_test_isic* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset isic --data_path *../data*``
change "data_path" and "exp_name" for your own useage. 

3. Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.  

For evaluation without train, you can directly download the model weights from the [checkpoints](https://drive.google.com/drive/folders/1Y6IBfUXt6_WwX0rM31v6XDJTV5baFw_t?usp=drive_link) trained on the the datasets mentioned in our paper for 50 epoch.

5. Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

In default, everything will be saved at `` ./logs/`` 

## Run on  your own dataset
It is simple to run BA-SAM on the other datasets. Just write another dataset class following which in `` ./dataset.py``. You only need to make sure you return a dict with 


     {
                 'image': A tensor saving images with size [C,H,W] for 2D image, size [C, H, W, D] for 3D data.
                 D is the depth of 3D volume, C is the channel of a scan/frame, which is commonly 1 for CT, MRI, US data. 
                 If processing, say like a colorful surgical video, D could the number of time frames, and C will be 3 for a RGB frame.

                 'label': The target masks. Same size with the images except the resolutions (H and W).

                 'p_label': The prompt label to decide positive/negative prompt. To simplify, you can always set 1 if don't need the negative prompt function.

                 'pt': The prompt. Should be the same as that in SAM, e.g., a click prompt should be [x of click, y of click], one click for each scan/frame if using 3d data.

                 'image_meta_dict': Optional. if you want save/visulize the result, you should put the name of the image in it with the key ['filename_or_obj'].

                 ...(others as you want)
     }


Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

### TODO LIST

- [x] Release base code and ckpt.
- [ ] Fix bugs
- [ ] More details

## Cite
If you use BA-SAM in your research, please use the following BibTeX entry. 
~~~
@article{song2024ba,
  title={BA-SAM: Scalable Bias-Mode Attention Mask for Segment Anything Model},
  author={Song, Yiran and Zhou, Qianyu and Li, Xiangtai and Fan, Deng-Ping and Lu, Xuequan and Ma, Lizhuang},
  journal={arXiv preprint arXiv:2401.02317},
  year={2024}
}
~~~
~~~
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
~~~
~~~
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
~~~



