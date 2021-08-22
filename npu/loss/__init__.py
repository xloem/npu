"""
:synopsis: Supported loss functions to use with NPU API for training.
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""

# CrossEntropyLoss = "CrossEntropyLoss"  #:
SparseCrossEntropyLoss = "SparseCrossEntropyLoss"  #:
MSELoss = "MSELoss"  #:
# CTCLoss = "CTCLoss"  #:
L1Loss = "L1Loss"  #:
# NLLLoss = "NLLLoss"  #:
SmoothL1Loss = 'SmoothL1Loss'  #: Huber Loss or smooth L1 loss
SigmoidBCELoss = 'SigmoidBCELoss'  #: Binary Cross Entropy Loss
# KLDivLoss = 'KLDivLoss'  #: Killback-Leibler divergence loss
# CosineEmbeddingLoss = 'CosineEmbeddingLoss' #: Pytorch equivalent Cosine Embedding Loss
SoftmaxCrossEntropyLoss = 'SoftmaxCrossEntropyLoss'  #: Softmax + CE Loss together
