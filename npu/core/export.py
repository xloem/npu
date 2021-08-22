import os
import re
import shutil
from io import BytesIO


from .common import getToken, check_model_type, npu_print, get
from .Task import Task
from .saving.saving import pytorch_str, mxnet_str, TF_str
from .web.urls import EXPORT_URL, MODELTYPE_URL
import tarfile
import tempfile


def export_task_result(task):
    if isinstance(task, Task):
        task = task.task_id
    task = Task(task, show=False)
    task.wait()
    return task.get_result()


def export(model, path, as_object, save_to_disk):
    if isinstance(model, Task):
        model = export_task_result(model)
    params = {"token": getToken()}
    check_model_type(model, params)
    file = export_api(params)
    name, model_type = model_type_api(params)
    if save_to_disk:
        filepath = os.path.join(path, name)
        with open(filepath, "wb") as file2:
            shutil.copyfileobj(file, file2)
            npu_print("Model exported to {}".format(filepath))
    if as_object:
        file.seek(0)
        return load_model(file, model_type)
    return name if save_to_disk else file


def export_api(params):
    response = get(EXPORT_URL, json=params, stream=True)
    if response.status_code == 200:
        file = BytesIO(response.content)
        file.seek(0)
        return file
    else:
        raise Exception("{}".format(response.content))


def model_type_api(params):
    response = get(MODELTYPE_URL, params=params, json=params)
    if response.status_code == 200:
        response = response.json()
        return response["name"], response["model_type"]
    else:
        raise Exception("{}".format(response.content))


def load_model(file, model_type):
    if model_type == pytorch_str:
        import torch
        import dill
        return torch.load(file, pickle_module=dill, map_location=torch.device("cpu"))
    elif model_type == mxnet_str:
        return load_mxnet(file)
    elif model_type == TF_str:
        return loadTF2Model(file)


def load_mxnet(file):
    import glob
    import json
    from mxnet import gluon
    with tempfile.TemporaryDirectory() as t_dir:
        extracted_dir = t_dir
        untar(file, extracted_dir)
        modelFile = glob.glob(extracted_dir + "/*.json")[0]
        paramFile = glob.glob(extracted_dir + "/*.params")[0]
        with open(modelFile) as json_file:
            data = json.load(json_file)
            inputs = [node["name"] for node in data["nodes"] if node["name"].startswith("data")]
            model = gluon.nn.SymbolBlock.imports(modelFile, inputs, paramFile)
    return model


def loadTF2Model(file):
    with tempfile.TemporaryDirectory() as t_dir:
        extracted_dir = t_dir
        # extracted_dir = untar(file)
        untar(file, extracted_dir)
        from tensorflow.keras.models import load_model
        model = load_model(extracted_dir, compile=False)
    return model


def untar(file, extracted_dir):
    with tarfile.open(fileobj=file, mode="r") as tar:
        # extracted_dir = file.split(".")[0] + "dir"
        tar.extractall(path=extracted_dir)
        return extracted_dir