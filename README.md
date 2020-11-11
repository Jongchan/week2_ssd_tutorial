# Week2 programming assignment - VOC07+12 and private dataset

## Environment settings

### Pre-requisite
Before we begin, the host server should meet the requirements below:
* Docker >= 19.03.8
* CUDA >= 10.2

### Download the data

```
# VOC2007
mkdir -p DATA/
sh data/scripts/VOC2007.sh DATA/

# VOC2012
mkdir -p DATA/
sh data/scripts/VOC2012.sh DATA/

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
### evaluation

```
python eval.py --trained_model DATA/weights/VOC.pth

## SAMPLE OUTPUT
VOC07 metric? Yes
AP for aeroplane = 0.8075
AP for bicycle = 0.8468
AP for bird = 0.7559
AP for boat = 0.7306
AP for bottle = 0.5035
AP for bus = 0.8608
AP for car = 0.8620
AP for cat = 0.8797
AP for chair = 0.6159
AP for cow = 0.8256
AP for diningtable = 0.7480
AP for dog = 0.8495
AP for horse = 0.8549
AP for motorbike = 0.8528
AP for person = 0.7905
AP for pottedplant = 0.5036
AP for sheep = 0.7605
AP for sofa = 0.7967
AP for train = 0.8728
AP for tvmonitor = 0.7652
Mean AP = 0.7741
~~~~~~~
Results:
0.807
0.847
0.756
0.731
0.504
0.861
0.862
0.880
0.616
0.826
0.748
0.849
0.855
0.853
0.790
0.504
0.760
0.797
0.873
0.765
0.774
~~~~~~~
```

## Task

### Task0: experiment logging and understand the code
* Use wandb for logging
* This includes training / validation loss, and validation performance.
* The base code has separate training/evaluation code, combine them into single code and iteratively do train/eval.
* Understand the items below and present in the final report:
  * How loss is implemented? (how are labels/bboxes used for loss computation? And how are they compared with the model outputs?)
  * How is the model output converted into bounding boxes?

### Task1: the effect of pre-trained weights
* Try with or without pre-trained weights.
* Observe the training / validation curve. If necessary, try using longer epochs.
* For more information on the effect of pre-trained weights, please refer to [Rethinking ImageNet Pre-training by Kaiming He](https://arxiv.org/abs/1811.08883)
  * It mentions the pre-training is only helpful in faster training convergence (in COCO dataset). BUT, is it still true in VOC dataset? Note that VOC is much smaller than COCO dataset. For further investivation on this issue, you may refer to [Rethinking Pre-training and Self-training](https://arxiv.org/abs/2006.06882)

### Task2: Tune other factors (at least 2)
* Resolution: current default resolution is 300 (specified in `data/config.py` as `min_dim`), what is the ultimate best resolution?
* Online hard sample mining (OHEM) or Focal Loss: there are easy / hard samples in the training. Online Hard Sample Mining (OHEM) is a popular technique to improve performance, and Focal Loss is a simple version of it. Try it, and see if there is some meaningful improvements.
* Backbone: VGG is a too simple backbone. How about using other backbones such as ResNet? (try with different sized backbones)

### Task3: Analyze how the model
* Any analysis on the model performance, apart from simple mAP. This may include some qualitative analysis. For example, calculate the model performance for large bboxes and small bboxes; why some classes have lower performance than others; etc.
