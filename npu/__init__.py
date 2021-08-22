"""
:synopsis: The main npu api
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""


from . import vision, optim, loss, metrics
from .core import DataLoader
from .version import __version__, _module
from . import core, optim, loss as _loss, metrics as _metrics


def api(token="", project="Default", verbosity=1, deployed=False):
    """Use this function to get access to the API, providing your token. All subsequents API calls will use this token.

        :param token: Token from dashboard. Set environment variable NPU_API_TOKEN to bypass
        :type token: str
        :param project: Project name to assign models/data/tasks to a certain project.
        :type project: str
        :param verbosity: Print level for client. Useful for debugging.
        :type verbosity: int
        :param deployed: In deployment mode or not
        :type deployed: bool
    """
    core.api(token, project, verbosity, deployed)


def compile(model, input_shape=None, library="", model_label="", scripts=None, input_kwargs=None, asynchronous=False):
    """Use this to upload and compile your model. Compatible frameworks are:

        * Pytorch
        * Tensorflow 2
        * Mxnet

        :param model: Original model to compile.
        :type model: Object from framework or filename(str) to the model. `Tensorflow` + `mxnet` models must be tarred if using filenames.
        :param input_shape: Input shape of model
        :type input_shape: List
        :param library: Library used
        :type library: str (pytorch, tf, mxnet)
        :param model_label: Label for model
        :type model_label: str, optional
        :param asynchronous: If call should run async or not. Default=`False`
        :type asynchronous: bool, optional.

        :return: compiled model.
    """
    if input_shape is None:
        input_shape = []
    if scripts is None:
        scripts = []
    if input_kwargs is None:
        input_kwargs = {}
    return core.compile(model, input_shape, library, model_label, scripts, input_kwargs, asynchronous)


def upload_data(data, name=""):
    """Use this to upload your data.

        :param data: Raw data to upload.
        :type data: (numpy, Pytorch Tensor, Mxnet NDArray, Tensorflow tf.Data)
        :param name: Name to be given to data.
        :type name: str, optional

        :return: data as id.
    """
    return core.upload_data(data, name)


def predict(model, data, input_kwargs=None, asynchronous=False, callback=None, scripts=None, **kwargs):
    """Perform a predict using a model. Default behaviour is synchronous.

        :param model: Model used to predict
        :type model: From :func:`npu.compile` or :func:`npu.train`. Id (str) or global :class:`npu.vision.models.Model` can be used.
        :param data: Data to be used for prediction
        :type data: numpy array
        :param asynchronous: If call should run async or not. Default=`False`. If you want to get the result back explicitly, call "get_result()" on returned value.
        :type asynchronous: bool, optional.
        :param callback: runs a callback function on results (asynchronous)
        :type callback: function
    """
    if scripts is None:
        scripts = []
    if input_kwargs is None:
        input_kwargs = {}
    return core.predict(model, data, input_kwargs, asynchronous, callback, scripts, **kwargs)


def train(model, train_data, val_data="", train_kwargs=None, val_kwargs=None, batch_size=32, epochs=1, optim=optim.SGD(), loss=_loss.SparseCrossEntropyLoss,
          metrics=None, trained_model_name="", asynchronous=False, callback=None, **kwargs):
    """Perform a train using a model. Default behaviour is synchronous.

        :param model: Model used to predict
        :type model: From :func:`npu.compile` or :func:`npu.train`. Id (str) or global :class:`npu.vision.models.Model`
        can be used.
        :param train_data: Training data in format of (x, y)
        :type train_data: numpy array
        :param val_data: Validation data in format of (x, y)
        :type val_data: numpy array
        :param batch_size: Batch size for training. Default=`32`
        :type batch_size: int, optional
        :param epochs: Epoch cycles for training. Default=`1`
        :type epochs: int, optional
        :param optim: Optimiser to use
        :type optim: :func:`npu.optim`
        :param loss: Loss function to use
        :type loss: Loss/function
        :param metrics: List of Metric functions to use
        :type metrics: Metric/function
        :param trained_model_name: Assignable name to be given to trained model
        :type trained_model_name: str, optional
        :param asynchronous: If call should run async or not. Default=`False`. If you want to get the result back
        explicitly, call "get_result()" on returned value.
        :type asynchronous: bool, optional.
        :param callback: runs a callback function on results (asynchronous)
        :type callback: function

    """
    if train_kwargs is None:
        train_kwargs = {}
    if val_kwargs is None:
        val_kwargs = {}
    assert type(batch_size) is int
    assert type(epochs) is int
    assert batch_size > 0
    assert epochs > 0
    if metrics is None:
        metrics = [_metrics.Accuracy]
    return core.train(model, train_data, val_data, train_kwargs, val_kwargs, batch_size, epochs, optim, loss, metrics, trained_model_name,
                      asynchronous, callback, **kwargs)


def export(model, path=".", as_object=True, save_to_disk=True):
    """Export a model to file. This will export it in the original format it is in. Global models will be exported as
    pytorch models.

        :param model: Model to export
        :type model: From :func:`npu.compile` or :func:`npu.train`. Id (str) or global :class:`npu.vision.models.Model` can be used.
        :param path: Path to where the model is saved to. Default is ".".
        :type path: str, optional
        :param as_object: If you wish to return the object back as an object in memory as a model in the respective framework
        :type as_object: bool, optional
        :param save_to_disk: By default, we always flush the file to disk. Set this to False to prevent flushing. Will
        also return the file as a file obj **only** if as_object is False.
        :type save_to_disk: bool, optional

    """
    return core.export(model, path, as_object, save_to_disk)


def export_task_result(task):
    """Export a Task to memory. Will generically work on all task types. Predictions return the explicit results,
    training/compile returns ids

        :param task: Task to export
        :type task: From :func:`npu.compile` or :func:`npu.train`. Id (str) or global :class:`npu.vision.models.Model` can be used.
    """
    return core.export_task_result(task)


def print(_):
    pass

