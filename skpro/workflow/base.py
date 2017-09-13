import abc
import hashlib
from sklearn.base import clone
import json


class Model:

    def __init__(self, instance, tuning=None, group=None, database={}):
        self.instance = clone(instance)
        self.group = group
        self.database = database
        if isinstance(tuning, dict) and len(tuning) == 0:
            # Convert empty dicts to None
            tuning = None

        self.tuning = tuning

    def __repr__(self):
        return repr(self.instance)

    def description(self):
        try:
            return self.instance.description()
        except:
            return repr(self.instance)

    def __getitem__(self, item):
        return self.database[item]

    def __setitem__(self, key, value):
        self.database[key] = value

    def clone(self):
        return Model(self.instance, self.tuning, self.group)

    def identifier(self, with_tuning=False):
        s = repr(self.instance)
        if with_tuning:
            s += json.dumps(self.tuning, sort_keys=True)

        return hashlib.md5(s.encode('utf-8')).hexdigest()


class Controller(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def identifier(self):
        pass

    @abc.abstractmethod
    def description(self):
        pass

    @abc.abstractmethod
    def run(self, model):
        pass


class View(metaclass=abc.ABCMeta):

    def __repr__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def parse(self, data):
        pass