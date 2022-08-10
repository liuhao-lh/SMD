# Improving Self-supervised Lightweight Model Learning via Hard-aware Metric Distillation
A PyTorch implementation of the paper SMD.


## Self-supervised Distillation on ImageNet
### Dependencies

If you don't have python 3 environment:
```
conda create -n SMD python=3.8
conda activate SMD
```
Then install the required packages:
```
pip install -r requirements.txt
```

Only **multi-gpu**, **DistributedDataParallel** training is supported; single-gpu or DataParallel training is not supported.

### Get teacher network

To pre-train a unsupervised ResNet-50 model on ImageNet, run:
```
python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your imagenet-folder with train and val folders]
```

Or you can download the pre-trained teacher model from [SimSiam](https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar).

### Unsupervised Distillation

To distill a ResNet-18 model on ImageNet in an 4-gpu machine, run:

```
python main_distill.py \
  -a resnet18 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --teacher_path [your pre-trained teacher model path] \
  [your imagenet-folder with train and val folders]
```

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features in an 4-gpu machine, run:
```
python main_lincls.py \
  -a resnet18 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained [your checkpoint path]/checkpoint_0099.pth.tar \
  --lars \
  [your imagenet-folder with train and val folders]
```

The above command uses LARS optimizer and a default batch size of 4096.

