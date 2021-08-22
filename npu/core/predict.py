import hashlib
import json

from .Dataset import Dataset
from .Task import Task, FAILURE
from .common import getToken, check_data, add_kwargs_to_params, check_model, determine_data, validate_model, \
    check_model_type, check_data_type, get_project, read_in_chunks, get_progress_bar_uploader, is_deployed, hash_file, \
    verbose_print, MID_VERBOSITY, get_response, post
from .saving.saving import save_data
from .web.urls import PREDICT_URL, BASE_URL, SYNCPREDICT_URL


def predict(model, data, input_kwargs, asynchronous=False, callback=None, scripts=[], **kwargs):
    if is_deployed():
        return deploy_predict(model, data, input_kwargs, scripts)
    check_model(model)
    inference_data = check_data(data)
    task_id = model.task_id if isinstance(model, Task) else ""
    resp = predict_api(model, inference_data, input_kwargs, scripts, task_id, **kwargs)
    task = Task(resp.text, callback)
    if not asynchronous:
        task.wait()
    return task


def predict_api(model, data, input_kwargs, scripts, task_id, **kwargs):
    data, data_name, start, end = determine_data(data)
    params = {"task_id": task_id, "data_start": start, "data_end": end,
              "data_name": data_name, "project": get_project(), "input_kwargs": input_kwargs, "scripts": scripts}
    check_model_type(model, params)
    check_data_type(data, "data", params)
    params = add_kwargs_to_params(params, **kwargs)
    response = post(PREDICT_URL, json=params)
    if response.status_code != 200:
        raise ValueError(response.text)
    return response


def deploy_predict(model, data, input_kwargs, scripts):
    verbose_print(f"Started deployed prediction using model {model}", MID_VERBOSITY)
    generic_file = False
    if isinstance(data, str):
        file = open(data, "rb")
        generic_file = True
    else:
        file = save_data(data)
    hash = hash_file(file)
    _json = {"hash": hash, "collection": 1, "project": get_project(), "modelId": model, "input_kwargs": input_kwargs,
             "scripts": scripts, "generic_file": generic_file}
    params = {"include_result": True}
    file.seek(0)
    # monitor = get_progress_bar_uploader(file=file, json=_json)
    # response = post(BASE_URL + "SyncPredict", data=monitor,
    #                          headers={'Content-Type': monitor.content_type}, params=params)

    response = post(SYNCPREDICT_URL, files={"file": file,
                                            "json": (None, json.dumps(_json),
                                                     'application/json')}, params=params)
    if response.status_code != 200:
        raise ValueError(response.text)
    response = get_response(response)
    task_state = response["state"]
    result = None
    if "result" in response:
        result = response["result"]
    if task_state == FAILURE:
        # raise ValueError(response.text)
        raise Exception("ERROR for prediction: {}".format(result))
    return result
    # return response
