# Adaptive Dynamic Filtering Network for Image Denoising (AAAI 2023)

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2211.12051)


## Environment

- Python 3.6 + Pytorch 1.0 + CUDA 10.1
- numpy
- skimage
- imageio
- cv2

## Get Started(Evaluation done by Mehdi Hamidi)

J'ai fait quelques modification pour faire fonctionner le code, voici les étapes à suivre pour évaluer le code en utilisant docker ! impossible de le faire sur roméo sauf si vous avez accèes aux droits root

```shell
# Création du conteneur
docker compose up --build
#Attach to the container using an extension or run it through CLI

# Once in the container follow this steps
cd Builder
sh make.sh

# Follow the steps below depending on your needs
```

Please make sure your machine has a GPU, which is required for the DCNv2 module.

## Training

### RGB image denoising

- Download [DIV2K](https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing) training data (800 training images) to train **ADFNet** or Download [DIV2K](https://drive.google.com/file/d/13wLWWXvFkuYYVZMMAYiMVdSA7iVEf2fM/view?usp=sharing)+[Flickr2K](https://drive.google.com/file/d/1J8xjFCrVzeYccD-LF08H7HiIsmi8l2Wn/view?usp=sharing)+[BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing)+[WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing) to train **ADFNet***.  


- for **ADFNet**

  Run `bash train_adfnet_n10.sh` or `bash train_adfnet_n30.sh`or `bash train_adfnet_n50.sh` or `bash train_adfnet_n70.sh`

- for **ADFNet***

  Run `bash train_adfnet-L_n50.sh`

### Gray image denoising

- Run `bash train_adfnet_n15.sh` or `bash train_adfnet_n25.sh`or `bash train_adfnet_n50.sh` 

### Real-world image denoising

- Download the SIDD-Medium dataset from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)

- Generate image patches

  `python generate_patches_SIDD.py --ps 256 --num_patches 300 --num_cores 10`

  and then place all image patches in `./datasets/sidd_patch`

- Download validation images of SIDD and place them in `./testsets/sidd/val`

- Install warmup scheduler

- Train your model with default arguments by running

  Run `bash train_adfnet.sh`

## Evaluation

Part of pre-trained models: [Google drive](https://drive.google.com/file/d/1wYw8mHSyxmutpHTahn_j4wjv_p4sJHeq/view?usp=share_link) or [Baidu cloud](https://pan.baidu.com/s/1eAbY3IBSLigkRJJfoQJ73A&pwd=1995)

### RGB image denoising

- cd ./ADFNet_RGB

- Download models and place it in ./checkpoints

- Download testsets ([CBSD68, Kodak24, McMaster](https://github.com/cszn/FFDNet/tree/master/testsets)) and place it in ./testsets
- Run `python test.py --save_images --chop `

### Gray image denoising

- cd ./ADFNet_Gray

- Download models and place it in ./checkpoints

- Download testsets ([BSD68, Urban100, Set12](https://github.com/cszn/FFDNet/tree/master/testsets)) and place it in ./testsets
- Run `python test.py --save_images --chop `

### Real-world image denoising

- cd ./ADFNet_Real

- Download the model and place it in ./pretrained_models

​	**Testing on SIDD datasets**

- Download sRGB validation [images](https://drive.google.com/drive/folders/1j5ESMU0HJGD-wU6qbEdnt569z7sM3479?usp=sharing) of SIDD and place them in ./datasets/sidd/val

- First, run `test_sidd_val_png.py --save_images --ensemble`, 

  and then run `evaluate_SIDD.m` to calculate the PSNR/SSIM value

- Download benchmark [BenchmarkNoisyBlocksSrgb.mat](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) of SIDD and place them in ./datasets/sidd/benchmark

- Run `test_sidd_benchmark_mat.py --save_images --ensemble`

​	**Testing on DND datasets**

- Download sRGB [images](https://drive.google.com/drive/folders/1-IBw_J0gdlM6AlqSm3Z7XWTXR-So4xzp?usp=sharing) of DND and place them in ./datasets/dnd/
- Run `test_dnd_png.py --save_images --ensemble`

## Citation

If you use ADFNet, please consider citing:

```
@article{shen2022adaptive,
  title={Adaptive Dynamic Filtering Network for Image Denoising},
  author={Shen, Hao and Zhao, Zhong-Qiu and Zhang, Wandi},
  booktitle={AAAI},
  year={2023}
}
```

**Acknowledgment**: This code is based on the [MIRNet](https://github.com/swz30/MIRNet) and [DAGL](https://github.com/jianzhangcs/DAGL) toolbox.
