import base64

import dill

from .Task import Task
from .common import getToken, check_data, add_kwargs_to_params, check_model, determine_data, check_model_type, \
    check_data_type, get_project, npu_print, post, HubDataset
from .web.urls import TRAIN_URL


def train(model, train_data, val_data, train_kwargs, val_kwargs, batch_size, epochs, optimiser, loss, metrics, trained_model_name, asynchronous,
          callback, **kwargs):
    check_model(model)
    train_data = check_data(train_data)
    val_data = check_data(val_data)
    task_id = model.task_id if isinstance(model, Task) else ""
    task = Task(train_api(model, train_data, val_data, train_kwargs, val_kwargs, batch_size, epochs, optimiser, loss,
                         metrics, trained_model_name, task_id, **kwargs).text, callback)
    if not asynchronous:
        task.wait()
        npu_print("Model finished training")
    return task


def train_api(model, train_data, val_data, train_kwargs, val_kwargs, batch_size, epochs, optimiser, loss, metrics, trained_model_name
             , task_id="", **kwargs):
    if not isinstance(trained_model_name, str):
        raise ValueError("Name given is not valid. Please supply a string.")
    train_data, train_name, train_start, train_end = determine_data(train_data)
    val_data, test_name, test_start, test_end = determine_data(val_data)
    if callable(loss):
        npu_print("Using custom loss function... {}".format(loss.__name__ if hasattr(loss, "__name__")
                                                                    else loss.__class__.__name__))
        loss = base64.urlsafe_b64encode(dill.dumps(loss)).decode()
    for i, m in enumerate(metrics):
        if callable(m) or mxnet_metric(m):
            npu_print("Serialising custom metric function... {}".format(m.__name__ if hasattr(m, "__name__")
                                                                    else m.__class__.__name__))
            metrics[i] = base64.urlsafe_b64encode(dill.dumps(m)).decode()
    params = {"loss": loss, "batch_size": batch_size, "epochs": epochs, "task_id": task_id,
              "train_start": train_start, "train_end": train_end, "test_start": test_start,
              "test_end": test_end, "train_name": train_name, "test_name": test_name,
              "trained_model_name": trained_model_name, "metrics": metrics, "project": get_project(),
              "optimiser": optimiser, "train_kwargs": train_kwargs, "val_kwargs": val_kwargs}
    check_model_type(model, params)
    check_data_type(train_data, "train", params)
    check_data_type(val_data, "test", params)
    params["train_hub_ds"] = isinstance(train_data, HubDataset)
    params["test_hub_ds"] = isinstance(val_data, HubDataset)
    params = add_kwargs_to_params(params, **kwargs)
    response = post(TRAIN_URL, json=params)
    if response.status_code != 200:
        raise ValueError(response.text)
    return response


def mxnet_metric(metric):
    try:
        from mxnet.metric import CustomMetric
        return isinstance(metric, CustomMetric)
    except:
        return False
