import hashlib
import json
import math
import os
import dill
import base64
from sys import exit
import requests
from bson import ObjectId
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization, hashes
from tqdm import tqdm
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from datetime import datetime

from .Model import Model
from .DataLoader import DataLoader
from .Dataset import Dataset
from .saving.saving import save_data, determine_model, TF_str, mxnet_str, pytorch_str
from .web.urls import TOKEN_URL, HASH_URL, UPLOAD_DATA_URL

VERBOSITY = 1
MIN_VERBOSITY = 1
MID_VERBOSITY = 2
FULL_VERBOSITY = 3

_token = ""
_project = ""
_deployed = False

utcnow = datetime.utcnow

with open(os.path.join(os.path.dirname(__file__), "pub_cred_key.pub"), "rb") as key_file:
    pub_key_encryption = serialization.load_pem_public_key(key_file.read())


# from SO
class bcolors:
    PURPLE = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\33[38;5;208m'


levels = {"WARNING": bcolors.ORANGE, "INFO": bcolors.PURPLE, "ERROR": bcolors.FAIL}

NEURO_AI_STR = f"[{bcolors.OKBLUE}Neuro Ai{bcolors.ENDC}]"


def api(token_, project_name, verbosity, deployed):
    global _token
    global _project
    global VERBOSITY
    global _deployed
    if token_ == "":
        token_ = os.environ.get("NPU_API_TOKEN", "")
    _token = token_
    VERBOSITY = verbosity
    verbose_print(f"Verbosity level set to {VERBOSITY}", MID_VERBOSITY)
    _deployed = deployed
    if _deployed:
        npu_print("DEPLOYMENT MODE")
    params = {"token": _token, "project_name": project_name}
    response = post(TOKEN_URL, json=params)
    if response.status_code == 200:
        npu_print("Token successfully authenticated")
        _project = response.json()
        npu_print(f"Using project: {project_name}")
        return response
    else:
        raise ValueError(response.text)
    # "API token not valid"


def getToken():
    return _token


def auth_header():
    return {"authorization": "Bearer " + getToken()}


def get_verbosity():
    return VERBOSITY


def get_project():
    return _project


def is_deployed():
    return _deployed


def get_response(response):
    try:
        return response.json()
    except Exception as e:
        raise ConnectionError("Invalid response received. Error: {}".format(response.text))


# https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def add_kwargs_to_params(params, **kwargs):
    params = {**params, **kwargs}
    return params


def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def check_model(model):
    from .Task import Task
    from .Model import Model
    if not isinstance(model, Task) and not isinstance(model, str) and not isinstance(model, Model):
        raise ValueError("Model is not a valid format. Please make sure you've compiled it first.")


def check_model_type(model, params):
    from .Task import Task
    if isinstance(model, Model):
        params["model_name"] = model.name
        params["model_attr"] = model.attr
    elif isinstance(model, str) and not ObjectId.is_valid(model):
        params["model_name"] = model
    elif model != "" and not isinstance(model, Task):
        params["modelId"] = model


def check_data_type(data, param_name, params):
    from .Task import Task
    if isinstance(data, Dataset):
        params[param_name + "_name"] = data.id
    elif isinstance(data, str) and not ObjectId.is_valid(data):
        params[param_name + "_name"] = data
    elif isinstance(data, HubDataset):
        params[param_name + "Id"] = data.hub_meta
    elif data != "" and not isinstance(data, Task):
        params[param_name + "Id"] = data
    params[f"{param_name}_hub_ds"] = isinstance(data, HubDataset)


def check_data(data, name=""):
    if not isinstance(name, str):
        raise ValueError("Name given is not valid. Please supply a string.")
    if isinstance(data, dict):
        return data
    try:
        import hub
        hub_meta = {}
        if hasattr(data, "dataset"):
            if hasattr(data, "indexes"):
                hub_meta["indexes"] = data.indexes
            if hasattr(data, "subpath"):
                hub_meta["subpath"] = data.subpath
            data = data.dataset
        if isinstance(data, hub.Dataset):
            encrypted_token = base64.b64encode(
                pub_key_encryption.encrypt(
                    json.dumps(data.token).encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(
                            algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))).decode()
            hub_meta = {"url": data.url, "schema": data.schema, "token": encrypted_token, **hub_meta}
            hub_meta = base64.b64encode(dill.dumps(hub_meta)).decode()
            return HubDataset(hub_meta)
    except Exception as e:
        # print(e)
        pass
    if isinstance(data, str) and (data.endswith(("npy", "npz")) or ObjectId.is_valid(data) or data == ""):
        return data
    elif isinstance(data, Dataset):
        return data
    elif isinstance(data, DataLoader):
        response = upload_data_loader(data, name)
    else:
        response = upload_data(data, name)
    status_code = response.status_code
    if status_code not in (204, 200, 201):
        raise ConnectionAbortedError("Data upload has not worked: {}".format(response.content))
    if status_code != 204:
        response = get_response(response)
    if isinstance(response, dict) and status_code == 200:
        message = response.get("message")
        npu_print(message)
        response = response["id"]
    return response


def slice_data(data):
    id = data["id"]
    start = data["indexes"]
    end = None
    if isinstance(start, slice):
        end = start.stop
        start = start.start
    return id, start, end


def gen(dl):
    for data_part in dl.numpy():
        yield save_data(data_part)


def create_callback(encoder):
    encoder_len = encoder.len
    bar = tqdm(desc=f"{NEURO_AI_STR} Uploading", unit="B", unit_scale=True, total=encoder_len, unit_divisor=1024)

    def callback(monitor):
        bar.n = monitor.bytes_read
        bar.refresh()
        if monitor.bytes_read == encoder_len:
            bar.close()

    return callback


def get_progress_bar_uploader(file, json):
    encoder = create_upload(file, json)
    callback = create_callback(encoder)
    monitor = MultipartEncoderMonitor(encoder, callback)
    return monitor


def create_upload(file, _json):
    return MultipartEncoder({
        'file': ('file', file, 'application/octet-stream', {'Content-Transfer-Encoding': 'binary'}),
        'json': (None, json.dumps(_json), 'application/json', {}),
    })


def upload_data_loader(dl, name=""):
    verbose_print("Hashing data locally...", MID_VERBOSITY)
    hash, size, length = dl.hash()
    params = {"token": getToken(), "hash": hash, "collection": 1, "chunked": True, "is_last": False, "size": size,
              "given_name": name, "input_shape": dl.shape, "project": get_project()}
    # params = {"token": getToken(), "hash": hash, "collection": 1, "size": size, "given_name": name}
    verbose_print("Checking if data is on servers...", MID_VERBOSITY)
    response = get(HASH_URL, params=params)
    if response.status_code == 200:
        verbose_print("Data already uploaded. Will not reupload.", MID_VERBOSITY)
        return response
    npu_print("Data not on servers. Starting to upload. Total size of data is {}".format(convert_size(size)))
    if length == 1:
        return upload_data(next(dl.numpy()), name)
    npu_print("{} chunks to upload...".format(length))
    for i, data_part in enumerate(dl.numpy()):
        verbose_print("Uploading chunk {} out of {}...".format(i + 1, length), MID_VERBOSITY)
        if i == length - 1:
            params["is_last"] = True
        file = save_data(data_part)
        monitor = get_progress_bar_uploader(file, params)
        response = post(UPLOAD_DATA_URL, data=monitor,
                        headers={'Content-Type': monitor.content_type})
    return response


def upload_data(data, name=""):
    verbose_print("Saving data locally...", FULL_VERBOSITY)
    generic_file = False
    if isinstance(data, str):
        file = open(data, "rb")
        generic_file = True
    else:
        file = save_data(data)
    verbose_print("Hashing...", FULL_VERBOSITY)
    hash = hashlib.md5()
    for piece in read_in_chunks(file):
        hash.update(piece)
    size = file.tell()
    hash = hash.hexdigest()
    verbose_print("Checking if data is on servers...", MID_VERBOSITY)
    params = {"token": getToken(), "hash": hash, "collection": 1, "given_name": name, "project": get_project(),
              "generic_file": generic_file}
    response = get(HASH_URL, params=params, json=params)
    if response.status_code == 200:
        verbose_print("Data already on servers. Returning result...", MID_VERBOSITY)
        file.close()
        return response
    npu_print("Data not found on servers. Total size of data is {}. Uploading now...".format(convert_size(size)))
    file.seek(0)

    monitor = get_progress_bar_uploader(file=file, json=params)
    response = post(UPLOAD_DATA_URL, data=monitor,
                    headers={'Content-Type': monitor.content_type})
    if isinstance(data, str):
        file.close()
    return response


def upload_sample(data, params):
    required = (len(data[0]) if isinstance(data, (tuple, list)) else len(data)) > 10
    if not required:
        return False
    data = [d[:10] for d in data] if isinstance(data, (tuple, list)) else data[:10]


def hash_file(file):
    hash = hashlib.md5()
    for piece in read_in_chunks(file):
        hash.update(piece)
        # break
    hash = hash.hexdigest()
    return hash


def validate_model(model, data):
    library = determine_model(model)
    if isinstance(data, str):
        return
    # data = convert_to_numpy(data)
    if library == pytorch_str:
        from torch import ones
    elif library == mxnet_str:
        from mxnet import nd
        ones = nd.ones
    elif library == TF_str:
        from numpy import ones
    else:
        return
        # raise ValueError("Cannot validate library: {} .".format(library))
    placeholder_data = ones(data.shape)
    model(placeholder_data)


def determine_data(data):
    start = end = None
    name = ""
    if isinstance(data, dict):
        data, start, end = slice_data(data)
    if isinstance(data, Dataset):
        name = data.id
        data = data
    return data, name, start, end


def npu_print(val, level="INFO"):
    log_str = f"{NEURO_AI_STR} {utcnow_formatted()} - [{levels[level]}{level}{bcolors.ENDC}]: {val}"
    print(f"{log_str}")


def verbose_print(str, verbosity):
    if VERBOSITY >= verbosity:
        npu_print(str)


def utcnow_formatted():
    return utcnow().strftime("%H:%M:%S")


def make_request(request_type_function, url, data, headers, json, params, **kwargs):
    if params is None:
        params = {}
    if json is None:
        json = {}
    if data is None:
        data = {}
    if headers is None:
        headers = {}
    try:
        response = request_type_function(url, data=data, headers={**headers, **auth_header()}, json=json,
                                         params=params, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as _:
        response = response.json()
        if "error" in response:
            npu_print(f"Error: {response['error']}", level="ERROR")
        elif "message" in response:
            npu_print(f"Error: {response['message']}", level="ERROR")
        raise Exception
    # exit(1)


def post(url, data=None, headers=None, json=None, params=None, **kwargs):
    return make_request(requests.post, url, data, headers, json, params, **kwargs)


def get(url, data=None, headers=None, json=None, params=None, **kwargs):
    return make_request(requests.get, url, data, headers, json, params, **kwargs)


class HubDataset:
    def __init__(self, hub_meta):
        self.hub_meta = hub_meta
