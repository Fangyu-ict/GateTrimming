##Gate Trimming: 
Implementation with PyTorch.
### Requirements
- Python == 3.6.6
- Pytorch == 1.6.0
- TorchVision == 0.7.0
- Ptflops ==  0.6.2
- TensorboardX == 2.1

### Pre-trained (Baseline) Models 
The pre-trained models with log files can be found in [Google Drive](https://drive.google.com/drive/folders/1Vgt5a2w-FvhQ1hhhfPKVl4s0e4bp-7TE?usp=sharing).
Please put the pre-trained models into the fold ``baseline``.
### Prune the VGG16 on CIFAR-10
```shell
python pruning_cifar.py --dataset cifar10 --arch vgg16 --pruning_rate 0.75 --pre_train True  
```
### Prune the ResNet-56 on CIFAR-10
```shell
python pruning_cifar.py --dataset cifar10 --arch resnet56 --pruning_rate 0.22 --pre_train True  
```

### Prune the ResNet-110 on CIFAR-10
```shell
python pruning_cifar.py --dataset cifar10 --arch resnet110 --pruning_rate 0.28 --pre_train True  
```

