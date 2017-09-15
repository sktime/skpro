import abc
import hashlib
import json

from sklearn.base import clone


class Model:

    def __init__(self, instance, tuning=None, group=None, database=None):
        self.instance = clone(instance)
        self.group = group
        if database is None:
            database = {}
        self.database = database
        if isinstance(tuning, dict) and len(tuning) == 0:
            tuning = None

        self.tuning = tuning

    def __repr__(self):
        return 'Model(' + repr(self.instance) + ')'

    def __str__(self):
        return 'Model(' + str(self.instance) + ')'

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
    def run(self, model):
        pass


class View(metaclass=abc.ABCMeta):

    def __repr__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def parse(self, data):
        pass