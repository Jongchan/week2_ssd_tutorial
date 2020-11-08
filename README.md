# Week2 programming assignment - VOC07+12 and private dataset

## Environment settings

### Pre-requisite
Before we begin, the host server should meet the requirements below:
* Docker >= 19.03.8
* CUDA >= 10.2

### Download the data

```
# VOC2007
mkdir -p DATA/VOC2007
sh data/scripts/VOC2007.sh DATA/VOC2007/

# VOC2012
mkdir -p DATA/VOC2012
sh data/scripts/VOC2012.sh DATA/VOC2012/

# Pre-trained weights for VGG16
mkdir -p DATA/weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth -P DATA/weights
```

### Build and run docker

```
cd docker
bash build_docker.sh
sh run_docker.sh
docker attach <DOCKER NAME>
```

## RUN

### train

```
# use default options
python train.py
```

## Task

### Task0: experiment logging
* Use wandb for logging
* This includes training / validation loss, and validation performance.
* The base code has separate training/evaluation code, combine them into single code and iteratively do train/eval.

### Task1: the effect of pre-trained weights
* Try with or without pre-trained weights.
* Observe the training / validation curve. If necessary, try using longer epochs.
* For more information on the effect of pre-trained weights, please refer to [Rethinking ImageNet Pre-training by Kaiming He](https://arxiv.org/abs/1811.08883)
  * It mentions the pre-training is only helpful in faster training convergence (in COCO dataset). BUT, is it still true in VOC dataset? Note that VOC is much smaller than COCO dataset. For further investivation on this issue, you may refer to [Rethinking Pre-training and Self-training](https://arxiv.org/abs/2006.06882)

### Task2: Tune other factors (at least 2)
* Resolution: current default resolution is 300 (specified in `data/config.py` as `min_dim`), what is the ultimate best resolution?
* Online hard sample mining (OHEM) or Focal Loss: there are easy / hard samples in the training. Online Hard Sample Mining (OHEM) is a popular technique to improve performance, and Focal Loss is a simple version of it. Try it, and see if there is some meaningful improvements.
* Backbone: VGG is a too simple backbone. How about using other backbones such as ResNet? (try with different sized backbones)
