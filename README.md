# My Modified Version of 3DGS
![Teaser image](assets/teaser.png)

## Todos

- [ ] Intergrate depth, alpha and segmentation rendering
- [ ] Support few-shot input

## Added Features
- Mantain rendered image names
- Log traininig with [wandb](https://wandb.ai/site)
- Depth, alpha and segmentation rendering
- Video rendering

## Commands

### Cloning the Repository
```
# SSH
git clone git@github.com:weijielyu/gaussian-splatting.git --recursive
```
or
```
# HTTPS
git clone https://github.com/weijielyu/gaussian-splatting --recursive
```

### Environment
```
conda create -n gs python=3.8 -y
conda activate gs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirement.txt
pip install submodules/simple-knn
```
- For vanilla Gaussian Splatting rendering:
```
pip install submodules/diff-gaussian-rasterization
```
- For depth, alpha rendering with confidence:
```
pip install submodules/diff-gaussian-rasterization-confidence
```
- For segmentation rendering:
```
pip install submodules/diff-gaussian-rasterization-gaga
```

### Colmap
```
python convert.py -s <location> --skip_matching [--resize] #If not resizing, ImageMagick is not needed
```

### Training
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
### Evaluation
```
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```


## Acknowledge
This repository is modified based on the following repositories. Thanks for their wonderful implementations!
```
https://github.com/graphdeco-inria/gaussian-splatting
https://github.com/VITA-Group/FSGS
https://github.com/lkeab/gaussian-grouping
```
