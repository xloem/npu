import os

import bson

from .Task import Task
from .common import MID_VERBOSITY, get_progress_bar_uploader, convert_size, \
    get_project, npu_print, verbose_print, get, post, get_response
from .saving import save_model
from .saving.saving import determine_model, hash_model
from .web.urls import RETRIEVE_MODEL_URL, HASH_URL, COMPILE_URL


def compile(model, input_shape, library, model_label, scripts, input_kwargs, asynchronous=False):
    if not isinstance(model_label, str):
        raise ValueError("Name given is not valid. Please supply a string.")
    if model_label != "" and model is None:
        npu_print(model_label)
        params = {"input_shape": input_shape, "label": model_label}
        response = get(RETRIEVE_MODEL_URL, params=params)
        if response.status_code == 200:
            return response
        else:
            raise LookupError("Model not found. " + str(response))
    if bson.ObjectId.is_valid(model) or isinstance(model, Task):
        return model
    if library == "":
        library = determine_model(model)
    hash = hash_model(model, library)
    params = {"input_shape": input_shape, "given_name": model_label, "hash": hash,
              "collection": 2, "modelType": library, "project": get_project(), "input_kwargs": input_kwargs,
              "scripts": scripts}
    response = get(HASH_URL, params=params, json=params)
    if response.status_code == 200:
        verbose_print("Model already on server. Returning result...", MID_VERBOSITY)
        response = get_response(response)
        if isinstance(response, dict):
            message = response.get("message")
            npu_print(message)
            response = response["id"]
        return response
    elif response.status_code != 204:
        raise ConnectionAbortedError("Checking hash not worked. {0}".format(response.content))
    else:
        verbose_print("Model not found on server.", MID_VERBOSITY)
        verbose_print("Saving model locally...", MID_VERBOSITY)
        file_path = save_model(model, library, input_shape)
        with open(file_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            verbose_print("Model saved locally.", MID_VERBOSITY)
            npu_print("Model file size is {}. Uploading model now...".format(convert_size(size)))
            task = Task(compile_api(file, params))
            task.cache = {"model": model}
            if not asynchronous:
                task.wait()
                npu_print("Model compiled successfully.")
            return task


def compile_api(file, params):
    file.seek(0)
    monitor = get_progress_bar_uploader(file, params)
    response = post(COMPILE_URL, data=monitor, headers={'Content-Type': monitor.content_type})
    response = get_response(response)
    return response

