"""
:synopsis: DataLoaders class wrapper
.. moduleauthor:: Naval Bhandari <naval@neuro-ai.co.uk>
"""
import hashlib
import itertools

from npu.core.saving.saving import convert_to_numpy


class DataLoader(object):

    def __init__(self, dl):
        self.dl = dl
        self.shape = []

    def __str__(self):
        return self.dl

    def __iter__(self):
        return iter(self.dl)

    def __len__(self):
        return len(self.dl)

    def numpy(self):
        for data in self.dl:
            yield convert_to_numpy(data)

    def hash(self):
        data_part = next(self.numpy())
        if not self.shape:
            self.shape = [s.shape for s in data_part]
        else:
            for i, s in enumerate(self.shape):
                s[0] += data_part[i][0]
        length = 1

        def get_size(byte_list):
            return sum(len(d) for d in byte_list)

        def get_bytes(data):
            return [d.tobytes() for d in data]

        d_bytes = get_bytes(data_part)
        hashes = [hashlib.md5(d) for d in d_bytes]
        size = get_size(d_bytes)
        for data_part in itertools.islice(self.numpy(), 1, None):
            length += 1
            d_bytes = get_bytes(data_part)
            size += get_size(d_bytes)
            for i, d in enumerate(d_bytes):
                hashes[i].update(d)
        _hash = hashlib.md5()
        for __hash in hashes:
            _hash.update(__hash.digest())
        return _hash.hexdigest(), size, length




