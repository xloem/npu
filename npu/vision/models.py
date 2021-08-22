"""
:synopsis: Global models available. All pre-trained version of the following models have been trained using ImageNet.
    Input image expected (3, H, W) where H and W are at least 224. Images expected to have a range between 0 and 1 and
    normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    Output vector of 1000 classes.
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""

from ..core import Model

alexnet = Model("alexnet")
""":alexnet: AlexNet model. Access pre-trained version `alexnet(pretrained=True)` """

vgg16 = Model("vgg16")
""":vgg16: VGG-16 model with batchnorm layers. Access pre-trained version `vgg16(pretrained=True)` """

vgg19 = Model("vgg19")
""":vgg19: VGG-19 model with batchnorm layers. Access pre-trained version `vgg19(pretrained=True)` """

resnet18 = Model("resnet18")
""":resnet18: ResNet-18 model. Access pre-trained version `resnet18(pretrained=True)` """

resnet34 = Model("resnet34")
""":resnet34: ResNet-34 model. Access pretrained version `resnet34(pretrained=True)` """

resnet50 = Model("resnet50")
""":resnet50: ResNet-50 model. Access pretrained version `resnet50(pretrained=True)` """

resnet101 = Model("resnet101")
""":resnet101: ResNet-101 model. Access pretrained version `resnet101(pretrained=True)` """

resnet152 = Model("resnet152")
""":resnet152: ResNet-152 model. Access pretrained version `resnet152(pretrained=True)` """

inception_v3 = Model("inception_v3")
""":inception_v3: Inception v3 model. Access pretrained version `inception_v3(pretrained=True)` """

resnext50_32x4d = Model("resnext50_32x4d")
""":resnext50_32x4d: ResNext-50 model. Access pretrained version `resnext50_32x4d(pretrained=True)` """

resnext101_32x8d = Model("resnext101_32x8d")
""":resnext101_32x8d: ResNext-101 model. Access pretrained version `resnext101_32x8d(pretrained=True)` """

mobilenet_v2 = Model("mobilenet_v2")
""":mobilenet_v2: MobileNet v2 model. Access pretrained version `mobilenet_v2(pretrained=True)` """

wide_resnet50_2 = Model("wide_resnet50_2")
""":wide_resnet50_2: Wide ResNet-50-2 model. Access pretrained version `wide_resnet50_2(pretrained=True)` """

wide_resnet101_2 = Model("wide_resnet101_2")
""":wide_resnet101_2: Wide ResNet-101-2 model. Access pretrained version `wide_resnet101_2(pretrained=True)` """

densenet161 = Model("densenet161")
""":densenet161: Densenet-161 model. Access pre-trained version `densenet161(pretrained=True)` """

googlenet = Model("googlenet")
""":googlenet: GoogleNet model. Access pre-trained version `googlenet(pretrained=True)` """

shufflenet_v2_x1_0 = Model("shufflenet_v2_x1_0")
""":shufflenet_v2_x1_0: ShuffleNetV2 model with 1.0x output channels. 
Access pre-trained version `shufflenet_v2_x1_0(pretrained=True)` """

mnasnet1_0 = Model("mnasnet1_0")
""":mnasnet1_0: MNASNet model. Access pre-trained version `mnasnet1_0(pretrained=True)` """



