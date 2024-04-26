# My Modified Version of 3DGS

## Todos

- [ ] Intergrate depth, alpha and segmentation rendering

## Added Features
- Rename rendered image names
- Log traininig with [wandb](https://wandb.ai/site)
- Video rendering
- Depth, alpha and segmentation rendering

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
conda env create --file environment.yml
conda activate gs
```


## Acknowledge
This repository is modified based on the following repositories. Thanks for their wonderful implementations!
```
https://github.com/graphdeco-inria/gaussian-splatting
https://github.com/VITA-Group/FSGS
https://github.com/lkeab/gaussian-grouping
https://github.com/ashawkey/diff-gaussian-rasterization
```