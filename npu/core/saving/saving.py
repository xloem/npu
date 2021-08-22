import glob
import hashlib
import os
import sys
import tarfile
from io import BytesIO
from collections.abc import Iterable
import numpy as np

pytorch_str = "pytorch"
mxnet_str = "mxnet"
TF_str = "TF"
TFLITE_str = "TFLITE"
FILE_str = "FILE"
onnx_str = "onnx"


def convert_to_numpy(data):
    if isinstance(data, (tuple, list)):
        return [framework_to_numpy(d) for d in data]
    else:
        return framework_to_numpy(data)


def save_model(model, library: str, input_shape: list):
    model_path = model
    if not isinstance(model, str):
        #     with tempfile.TemporaryDirectory() as t_dir:
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        model_path = "tmp/tmp_model"
        if library == pytorch_str:
            from torch import save
            import dill
            model_path += ".pt"
            try:
                from pytorch_lightning import LightningModule
                if isinstance(model, LightningModule):
                    try:
                        import cloudpickle
                        dill = cloudpickle
                    except ModuleNotFoundError as _:
                        from ..common import npu_print
                        npu_print("To use pytorch lightning, "
                                  "please install cloudpickle>=1.6.0 using pip install cloudpickle -U", level="ERROR")
                        sys.exit(1)
            except ModuleNotFoundError as _:
                pass
            save(model, model_path, pickle_module=dill)
        elif library == "keras":
            model_path += ".h5"
            import keras
            with keras.backend.get_session().graph.as_default():
                model.save_model(model_path)
        elif library == mxnet_str:
            model_path += ".tar"
            model.export(model_path)
            with tarfile.open(model_path, "w") as tar:
                jsonname = model_path + "-symbol.json"
                paramname = model_path + "-0000.params"
                tar.add(jsonname, arcname=os.path.basename(jsonname))
                tar.add(paramname, arcname=os.path.basename(paramname))
        elif library == TF_str:
            model_path += ".tar"
            model.save(model_path + "dir")
            # save_model(model, model_path + "dir", include_optimizer=True, save_format='tf')
            with tarfile.open(model_path, "w") as tar:
                for file in glob.glob(model_path + "dir/*"):
                    tar.add(file, arcname=os.path.basename(file))
        elif library == "TF1":
            model_path += ".pb"
        elif library == onnx_str:
            model_path += ".onnx"
            try:
                from onnx import ModelProto, save
                if isinstance(model, ModelProto):
                    save(model, model_path)
            except ModuleNotFoundError as _:
                from ..common import npu_print
                npu_print("To use onnx install onnx", level="ERROR")
                sys.exit(1)
            try:
                from torch import nn, ones, onnx
                if isinstance(model, nn.Module):
                    input_name = "input"
                    with open(model_path, "wb") as f:
                        if len(input_shape) > 0 and isinstance(input_shape[0], Iterable):
                            input_samples = {f"{input_name}{i}": ones(input_shape[i]) for i in range(len(input_shape))}
                        else:
                            input_samples = {input_name: ones(input_shape)}
                        dynamic_axes_inputs = {name: {0: 'batch_size'} for name in input_samples}
                        onnx.export(model, tuple(input_samples.values()), f=f,
                                    export_params=True,
                                    do_constant_folding=True,
                                    input_names=list(input_samples.keys()),
                                    output_names=['output'],
                                    dynamic_axes={**dynamic_axes_inputs, 'output': {0: 'batch_size'}})
            except ModuleNotFoundError as _:
                from ..common import npu_print
                npu_print("To use pytorch lightning with onnx backend install onnx", level="ERROR")
                sys.exit(1)
        else:
            raise ValueError("Model type: " + str(library) + " not defined")
    return model_path

    # if model_type is ModelType.ONNX:
    #     onnx.save(model, file)
    # elif model_type is ModelType.TF1:
    #     pass
    # tf.saved_model.save(model, "./dd.pb")
    # tf.compat.v1.saved_model.save(model, "tmp")
    # raise ValueError("Tensorflow 1 incompatible. Please use .pb file directly if using Tensorflow 1.")


def utf_str(obj):
    return str(obj).encode("utf-8")


def hash_model(model, library):
    hash = hashlib.md5()
    if isinstance(model, str):
        with open(model, "rb") as file:
            return hashlib.md5(file.read()).hexdigest()
    elif library == pytorch_str:
        hash.update(utf_str(model))
        for p in model.parameters():
            hash.update(utf_str(p))
    elif library == mxnet_str:
        from mxnet import sym
        x = sym.var('data')
        x = model(x).tojson()
        hash.update(utf_str(x))
        for param in sorted(model.collect_params().items()):
            hash.update(utf_str(param[1].data()))
    elif library == TF_str:
        try:
            hash.update(utf_str(model.to_json()))
            for param in model.trainable_variables:
                hash.update(utf_str(param))
        except NotImplementedError:
            return ""
    elif library == onnx_str:
        return ""
    else:
        raise ValueError("Cannot hash... Model type: {} not defined.".format(library))
    return hash.hexdigest()


def framework_to_numpy(data):
    try:
        if isinstance(data, np.ndarray):
            return data
    except:
        pass
    try:
        from torch import Tensor
        if isinstance(data, Tensor):
            return data.numpy()
    except:
        pass
    try:
        from tensorflow import is_tensor, make_ndarray
        if is_tensor(data):
            return make_ndarray(data)
    except:
        pass
    try:
        from mxnet import nd
        if isinstance(data, nd.NDArray):
            return data.asnumpy()
    except:
        pass
    raise ValueError("Could not determine framework data is from. Try pass in numpy array directly.")


def save_data(data):
    file = BytesIO()
    if isinstance(data, (tuple, list)):
        dict_data = {str(i): data[i] for i in range(0, len(data))}
        np.savez(file, **dict_data)
    else:
        np.savez(file, data)
    file.seek(0)
    return file


def determine_model(model):
    try:
        from torch.nn import Module
        if isinstance(model, Module):
            return pytorch_str
    except:
        pass
    try:
        from tensorflow.keras import Model
        if isinstance(model, Model):
            return TF_str
    except:
        pass
    try:
        from mxnet.gluon.nn import Block
        if isinstance(model, Block):
            return mxnet_str
    except:
        pass
    try:
        from onnx import ModelProto
        if isinstance(model, ModelProto):
            return onnx_str
    except:
        pass
    if isinstance(model, str):
        if model.endswith(".pt"):
            return pytorch_str
    raise ValueError("Could not determine framework model is from. Please specify explicitly.")
