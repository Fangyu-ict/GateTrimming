"""The models subpackage contains definitions for the following models
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a models with random weights by calling its constructor:
.. code:: python
    import models
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110,resnet_reconstruct

from .densenet import densenet40, densenet40_reconstrcut

from .vgg import vgg11, vgg13, vgg16, vgg19,vgg11_imagenet, vgg13_imagenet, vgg16_imagenet, vgg19_imagenet,\
    vgg11_bn_imagenet, vgg13_bn_imagenet, vgg16_bn_imagenet, vgg19_bn_imagenet, vgg_reconstruct,cfg