# NgeNet: Neighborhood-aware Geometric Encoding Network for Point Cloud Registration

### [Neighborhood-aware Geometric Encoding Network for Point Cloud Registration](https://arxiv.org/pdf/2201.12094.pdf)

[Lifa Zhu](https://github.com/zhulf0804), [Haining Guan](https://github.com/qsisi), Changwei Lin, [Renmin Han*](https://scholar.google.com/citations?user=5PEiWnkAAAAJ&hl=zh-CN&oi=ao)

## Updates

- **2022-03-04** The code is publicly avaliable here.
- **2022-01-31** The paper is avaliable at [arXiv](https://arxiv.org/abs/2201.12094).

## Environments

- All experiments were run on a RTX 3090 GPU with an  Intel 8255C CPU at 2.50GHz CPU.  Dependencies can be found in `requirements.txt`.

- Compile python bindings

    ```
    # Compile

    cd NgeNet/cpp_wrappers
    sh compile_wrappers.sh
    ```

## 0. Pretrained weights (Optimal)

Download pretrained weights for 3DMatch, 3DLoMatch, Odometry KITTI and MVP-RG from [GoogleDrive](https://drive.google.com/drive/folders/1JDn6zQfLdZfAVVboXRrrrCVRo48pRjyW?usp=sharing) or [BaiduDisk](https://pan.baidu.com/s/18G_Deim1UlSkY8wWoOiwnw) (pwd: `vr9g`).

## 1. 3DMatch and 3DLoMatch

### dataset

We adopt the 3DMatch and 3DLoMatch provided from [PREDATOR](https://github.com/overlappredator/OverlapPredator), and download it [here](https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip) [**5.17G**].
Unzip it, then we should get the following directories structure:

``` 
| -- indoor
    | -- train (#82, cats: #54)
        | -- 7-scenes-chess
        | -- 7-scenes-fire
        | -- ...
        | -- sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_4
    | -- test (#8, cats: #8)
        | -- 7-scenes-redkitchen
        | -- sun3d-home_md-home_md_scan9_2012_sep_30
        | -- ...
        | -- sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

### train

```
## Reconfigure configs/threedmatch.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the 3dMatch dataset.

cd NgeNet
python train.py configs/threedmatch.yaml

# note: The code `torch.cuda.empty_cache()` in `train.py` has some impact on the training speed.
# You can remove it or change its postion according to your GPU memory. 
```

### evaluate and visualize

```
cd NgeNet

python eval_3dmatch.py --benchmark 3DMatch --data_root your_path/indoor --checkpoint your_path/3dmatch.pth --saved_path work_dirs/3dmatch [--vis] [--no_cuda]

python eval_3dmatch.py --benchmark 3DLoMatch --data_root your_path/indoor --checkpoint your_path/3dmatch.pth --saved_path work_dirs/3dlomatch [--vis] [--no_cuda]
```

## 2. Odometry KITTI

### dataset

Download odometry kitti [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with `[velodyne laser data, 80 GB]` and `[ground truth poses (4 MB)]`, then unzip and organize in the following format.

```
| -- kitti
    | -- dataset
        | -- poses (#11 txt)
        | -- sequences (#11 / #22)
    | -- icp (generated automatically when training and testing)
        | -- 0_0_11.npy
        | -- ...
        | -- 9_992_1004.npy
```

### train

```
## Reconfigure configs/kitti.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the Odometry KITTI.

cd NgeNet
python train.py configs/kitti.yaml
```

### evaluate and visualize

```
cd NgeNet
python eval_kitti.py --data_root your_path/kitti --checkpoint your_path/kitti.pth [--vis] [--no_cuda]
```

## 3. MVP-RG

### dataset

Download MVP-RG dataset [here](https://mvp-dataset.github.io/MVP/Registration.html), then organize in the following format.

```
| -- mvp_rg
    | -- MVP_Train_RG.h5
    | -- MVP_Test_RG.h5
```

### train

```
## Reconfigure configs/mvp_rg.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the MVP-RG.

python train.py configs/mvp_rg.yaml

# note: The code `torch.cuda.empty_cache()` in `train.py` has some impact on the training speed.
# You can remove it or change its postion according to your GPU memory. 
```

### evaluate and visualize

```
python eval_mvp_rg.py --data_root your_path/mvp_rg --checkpoint your_path/mvp_rg.pth [--vis] [--no_cuda]
```

## Citation

```
@article{zhu2022neighborhood,
  title={Neighborhood-aware Geometric Encoding Network for Point Cloud Registration},
  author={Zhu, Lifa and Guan, Haining and Lin, Changwei and Han, Renmin},
  journal={arXiv preprint arXiv:2201.12094},
  year={2022}
}
```

## Acknowledgements

Thanks for the open source code [OverlapPredator](https://github.com/overlappredator/OverlapPredator), [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), [KPConv.pytorch](https://github.com/XuyangBai/KPConv.pytorch), [FCGF](https://github.com/chrischoy/FCGF), [D3Feat.pytorch](https://github.com/XuyangBai/D3Feat.pytorch), [MVP_Benchmark](https://github.com/paul007pl/MVP_Benchmark) and [ROPNet](https://github.com/zhulf0804/ROPNet).