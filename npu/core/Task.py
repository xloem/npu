import time

from tqdm import tqdm

from .common import npu_print, NEURO_AI_STR, get_response, get
from .web.urls import TASK_STATUS_URL
from threading import Thread

FAILURE = "FAILURE"
PENDING = "PENDING"
COMPLETE = "COMPLETE"
STOPPED = "STOPPED"

TASK_DONE_LIST = (FAILURE, COMPLETE, STOPPED)

bar_suffix = NEURO_AI_STR + " {desc}: {percentage:.1f}%|{bar}| remaining: {remaining} || elapsed: {elapsed} "
dash_str = "Started {0}. View status at https://dashboard.getneuro.ai/task?id={1}"
one_hundred_percent = 100


class Task:

    def __init__(self, task_id, callback=None, show=True):
        self.task_id = task_id
        self.url = TASK_STATUS_URL + self.task_id
        self.task_result = ""
        self.task_type = ""
        self.progress = 0
        self.task_state = PENDING
        self.callback = callback
        self.cache = None
        self._logs = {}
        self.prints = []
        self.params = {"include_result": False}
        if self.callback:
            t = Thread(target=self.callback_thread)
            t.start()
        if show:
            self.update()
            npu_print(dash_str.format(self.task_type.lower(), task_id))
            
    def wait(self):
        with tqdm(desc=self.task_type, total=one_hundred_percent,
                  bar_format=bar_suffix) as bar:
            while not self.finished():
                time.sleep(0.1)
                [bar.write(log) for log in self.prints]
                self.prints = []
                bar.n = self.progress * one_hundred_percent
                bar.refresh()
            if self.task_state == FAILURE:
                self.update(include_result=True)
                npu_print(f"Error for task {self.task_id}: {self.task_result}", level="ERROR")
                raise Exception
                # exit(1)
            if self.task_state == STOPPED:
                npu_print("Task has been stopped.")
                return
            bar.n = one_hundred_percent

    def callback_thread(self):
        self.get_result()
        self.callback(self)

    def get_result(self):
        self.wait()
        self.update(include_result=True)
        return self.task_result

    def update(self, include_result=False):
        self.params["include_result"] = include_result
        response = get(self.url, params=self.params)
        response = get_response(response)
        self.task_state = response["state"]
        self.task_type = response["taskType"]
        self.progress = response["progress"]
        if "result" in response:
            self.task_result = response["result"]
        if "metrics" in response:
            self._logs = response["metrics"]
        if "print" in response:
            self.prints = response["print"]

    def __str__(self):
        return str(self.get_result())

    def finished(self):
        self.update()
        return self.task_state in TASK_DONE_LIST

    def logs(self):
        self.get_result()
        return self._logs


