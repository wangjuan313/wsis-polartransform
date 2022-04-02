# Polar Transformation Based Multiple Instance Learning Assisting Weakly Supervised Image Segmentation With Loose Bounding Box Annotations

This project hosts the codes for the implementation of the paper **Polar Transformation Based Multiple Instance Learning Assisting Weakly Supervised Image Segmentation With Loose Bounding Box Annotations** [[arxiv](https://arxiv.org/abs/2203.06000)].

# Dataset preprocessing

Download [Promise12](https://promise12.grand-challenge.org/) dataset, and put it on the "data/prostate" folder.

Run the following codes for preprocessing:

```bash
# trainig and valid subsets for promise12 dataset
python preprocess/slice_promise_train_val.py
python preprocess/slice_promise_augment_train_val.py
```

# Training

```bash
# The following experiments include Generalized MIL (exp_no=1), 
# Polar transformation based MIL (exp_no=2,3), 
# the proposed approach with weighted alpha-softmax approaximation (exp_no=4,...,10),
# the proposed approach with weighted alpha-quasimax approaximation (exp_no=11,...,17),
CUDA_VISIBLE_DEVICES=0 python tools/train_promise_unetwithbox_polartransform.py --n_exp exp_no

```

```bash
# Dice validation results for promise12 dataset, exp_no=1,2,...,50
CUDA_VISIBLE_DEVICES=0 python tools/valid_promise_unetwithbox_polartransform.py --n_exp exp_no
```

# Performance summary

```bash
python tools/report_promise_unetwithbox_polartransform.py
```

# Center visualization

```bash```
# exp_no = 1 or 2
python tools/plot_promise_center_polartransform.py --n_exp exp_no
```

![](C:\Users\wangj\Desktop\tmp\miccai2022\visualization\0_Case16_0_00.png_8.gif) <img src="file:///C:/Users/wangj/Desktop/tmp/miccai2022/visualization/0_Case25_0_01.png_8.gif" title="" alt="" data-align="inline"> ![](C:\Users\wangj\Desktop\tmp\miccai2022\visualization\0_Case31_0_01.png_7.gif)

## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{wang2021bounding,
  title={Bounding Box Tightness Prior for Weakly Supervised Image Segmentation},
  author={Wang, Juan and Xia, Bin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={526--536},
  year={2021},
  organization={Springer}
}
@article{wang2022polar,
  title={Polar Transformation Based Multiple Instance Learning Assisting Weakly Supervised Image Segmentation With Loose Bounding Box Annotations},
  author={Wang, Juan and Xia, Bin},
  journal={arXiv preprint arXiv:2203.06000},
  year={2022}
}
```


