# NPU Python Client Package

To install, run:

`pip install npu`

To use in code:

`import npu` 


This library has four core functions. `npu.api, npu.compile, npu.train, npu.predict` Here is an example script of them in use. 

```
import npu
from npu.vision.models import resnet18
from npu.vision.datasets import CIFAR10


npu.api(API_KEY)

model = npu.train(resnet18(pretrained=True),
                         train_data=CIFAR10.train,
                         val_data=CIFAR10.val,
                         loss=npu.loss.CrossEntropyLoss,
                         optim=npu.optim.SGD(lr=0.01),
                         batch_size=128,
                         epochs=2)

output = npu.predict(model, data)

```

Full documentation is [here](https://dashboard.neuro-ai.co.uk/docs/)
