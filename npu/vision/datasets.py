"""
:synopsis: Global datasets available
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""

from ..core import Dataset

CIFAR10 = Dataset("cifar10", 40000, 50000, 60000)
""":CIFAR10: CIFAR10 dataset. Contains training/validation/test subsets of data. Accessed using 
`CIFAR10.train/val/test` properties. Can also get direct subset using slices. 
CIFAR10.train and CIFAR10.val contain the training set of the dataset with a split of 80% and 20% respectively.
CIFAR10.test contains 10000 samples from the non-training set of the dataset. 
"""

CIFAR100 = Dataset("cifar100", 40000, 50000, 60000)
""":CIFAR100: CIFAR100 dataset. Contains training/validation/test subsets of data. Accessed using 
`CIFAR100.train/val/test` properties. Can also get direct subset using slices. 
CIFAR100.train and CIFAR100.val contain the training set of the dataset with a split of 80% and 20% respectively.
CIFAR100.test contains 10000 samples from the non-training set of the dataset. 
"""

MNIST = Dataset("mnist", 48000, 60000, 70000)
""":MNIST: MNIST dataset. Contains training/validation/test subsets of data. Accessed using 
`MNIST.train/val/test` properties. Can also get direct subset using slices. 
MNIST.train and MNIST.val contain the training set of the dataset with a split of 80% and 20% respectively.
MNIST.test contains 10000 samples from the non-training set of the dataset. 
"""

KMNIST = Dataset("kmnist", 48000, 60000, 70000)
""":KMNIST: KMNIST dataset. Contains training/validation/test subsets of data. Accessed using 
`KMNIST.train/val/test` properties. Can also get direct subset using slices. 
KMNIST.train and KMNIST.val contain the training set of the dataset with a split of 80% and 20% respectively.
KMNIST.test contains 10000 samples from the non-training set of the dataset. 
"""

QMNIST = Dataset("qmnist", 48000, 60000, 120000)
""":QMNIST: QMNIST dataset. Contains training/validation/test subsets of data. Accessed using 
`QMNIST.train/val/test` properties. Can also get direct subset using slices. 
QMNIST.train and QMNIST.val contain the training set of the dataset with a split of 80% and 20% respectively.
QMNIST.test contains the 60000 samples from the non-training set of the dataset. 
"""

FashionMNIST = Dataset("fashion_mnist", 48000, 60000, 70000)
""":FashionMNIST: FashionMNIST dataset. Contains training/validation/test subsets of data. Accessed using 
`FashionMNIST.train/val/test` properties. Can also get direct subset using slices. 
FashionMNIST.train and FashionMNIST.val contain the training set of the dataset with a split of 80% and 20% respectively.
FashionMNIST.test contains the 10000 samples from the non-training set of the dataset. 
"""

# CocoDetection = Dataset("coco_detection", 118287, 123287, 163957)
# """:CocoDetection: CocoDetection dataset. Contains training/validation/test subsets of data. Accessed using
# `CocoDetection.train/val/test` properties. Can also get direct subset using slices.
# CocoDetection.train contains the 118287 samples from the train set of the dataset.
# CocoDetection.val contains the 5000 samples from the val set of the dataset.
# CocoDetection.test contains the 40670 samples from the test set of the dataset.
# """

