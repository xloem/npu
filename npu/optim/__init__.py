"""
:synopsis: Optimisers to use with NPU API for training
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""


def SGD(lr=0.001, momentum=0):
    """SGD Optimiser.

          :param lr: Learning rate
          :type lr: float
          :param momentum: Momentum
          :type momentum: float
      """
    return {
        "optimiser": "SGD",
        "opt_args": {
            "lr": lr,
            "momentum": momentum
        }}


def RMS(lr=0.001, decay=0):
    """RMS Optimiser.

              :param lr: Learning rate
              :type lr: float
              :param decay: Decay
              :type decay: float
          """
    return {
        "optimiser": "RMS",
        "opt_args": {
            "lr": lr,
            "decay": decay
        }}


def Adam(lr=0.001, weight_decay=0):  # , beta1=0.9, beta2=0.999, epsilon=1e-08):
    """Adam Optimiser.

              :param lr: Learning rate
              :type lr: float
              :param weight_decay: Weight decay
              :type weight_decay: float, optional
              # :param beta1: Beta 1
              # :type beta1: float
              # :param beta2: Beta 2
              # :type beta2: float
              # :param epsilon: Epsilon
              # :type epsilon: float
          """
    return {
        "optimiser": "Adam",
        "opt_args": {
            "lr": lr,
            "weight_decay": weight_decay
            #             "beta1": beta1,
            #             "beta2": beta2,
            #             "epsilon": epsilon,
        }}
